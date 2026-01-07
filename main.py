"""
Nova GetChat - 合并转发上下文注入插件
作者: Nova for 辉宝主人
功能:
  1. 检测合并转发消息(直接发送/回复引用)
  2. 提取转发内容并格式化
  3. 使用LLM对转发内容进行摘要（可选）
  4. 自动注入到LLM上下文中
"""

import datetime
import traceback
import base64
import uuid
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest, LLMResponse, Provider
from astrbot.api import logger, AstrBotConfig
from astrbot.api.message_components import Plain, Image, Forward, Reply
from astrbot.api.platform import MessageType
import astrbot.api.message_components as Comp

try:
    from astrbot.core.utils.io import download_image_by_url
except ImportError:
    async def download_image_by_url(url: str) -> str:
        return url

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


@register("nova-getchat", "Nova", "合并转发上下文注入插件 - 自动解析转发消息并注入AI上下文", "1.0.0")
class NovaGetChatPlugin(Star):
    """Nova合并转发上下文注入插件"""
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 会话历史存储
        self.session_forwards = defaultdict(list)
        """存储每个会话的转发消息内容"""
        
        # 基础配置读取
        self.enable_forward_analysis = bool(self.get_cfg("enable_forward_analysis", True))
        self.forward_template = self.get_cfg("forward_template",
            "【转发消息记录 - 共{count}条】\n━━━━━━━━━━━━━━━━━━━━\n{content}\n━━━━━━━━━━━━━━━━━━━━")
        self.message_template = self.get_cfg("message_template", "[{sender}] {text}")
        self.image_mode = self.get_cfg("image_mode", "placeholder")
        self.enable_nested = bool(self.get_cfg("enable_nested_forward", True))
        self.max_nested_depth = int(self.get_cfg("max_nested_depth", 3))
        self.max_messages = int(self.get_cfg("max_messages", 50))
        self.inject_position = self.get_cfg("inject_position", "before_user")
        
        # LLM摘要配置
        self.enable_llm_summary = bool(self.get_cfg("enable_llm_summary", False))
        self.summary_provider_id = self.get_cfg("summary_provider_id", "")
        self.summary_prompt = self.get_cfg("summary_prompt",
            "请阅读以下聊天记录，并生成一份简洁的摘要。要求：\n1. 保留关键信息和重要观点\n2. 标注主要发言者\n3. 如有讨论结论，请明确指出\n4. 摘要控制在300字以内\n\n聊天记录：\n{content}")
        self.summary_inject_template = self.get_cfg("summary_inject_template",
            "【转发消息摘要 - 原始{count}条消息】\n{summary}")
        self.keep_original_on_fail = bool(self.get_cfg("keep_original_on_summary_fail", True))
        
        logger.info("[Nova-GetChat] 插件已初始化")
        logger.info(f"[Nova-GetChat] 合并转发分析: {'已启用' if self.enable_forward_analysis else '已禁用'}")
        logger.info(f"[Nova-GetChat] 图片处理模式: {self.image_mode}")
        logger.info(f"[Nova-GetChat] 嵌套转发: {'已启用' if self.enable_nested else '已禁用'}")
        logger.info(f"[Nova-GetChat] LLM摘要: {'已启用' if self.enable_llm_summary else '已禁用'}")
    
    def get_cfg(self, key: str, default=None):
        """获取配置项"""
        return self.config.get(key, default)
    
    async def _detect_forward_message(self, event: AstrMessageEvent) -> Optional[str]:
        """检测合并转发消息并返回forward_id
        
        支持两种场景:
        1. 直接发送的合并转发
        2. 回复引用的合并转发
        """
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            logger.debug("[Nova-GetChat] 非aiocqhttp平台，跳过转发检测")
            return None
        
        # 场景1: 直接发送的合并转发
        for seg in event.message_obj.message:
            if isinstance(seg, Forward):
                logger.info("[Nova-GetChat] 检测到直接发送的合并转发")
                return seg.id
        
        # 场景2: 回复引用的合并转发
        reply_seg = None
        for seg in event.message_obj.message:
            if isinstance(seg, Reply):
                reply_seg = seg
                break
        
        if reply_seg:
            try:
                client = event.bot
                original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                
                if original_msg and 'message' in original_msg:
                    original_chain = original_msg['message']
                    if isinstance(original_chain, list):
                        for segment in original_chain:
                            if isinstance(segment, dict) and segment.get("type") == "forward":
                                logger.info("[Nova-GetChat] 检测到回复引用的合并转发")
                                return segment.get("data", {}).get("id")
            except Exception as e:
                logger.error(f"[Nova-GetChat] 获取回复消息失败: {e}")
        
        return None
    
    async def _extract_forward_content(self, event: AstrMessageEvent, forward_id: str, 
                                        depth: int = 0) -> Tuple[List[Dict[str, Any]], List[str]]:
        """提取合并转发消息内容
        
        返回: (消息列表[{sender, text, images}], 图片URL列表)
        """
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return [], []
        
        if depth > self.max_nested_depth:
            logger.warning(f"[Nova-GetChat] 嵌套深度超过限制 {self.max_nested_depth}")
            return [], []
        
        try:
            client = event.bot
            forward_data = await client.api.call_action('get_forward_msg', id=forward_id)
            messages = forward_data.get("messages", [])
            
            extracted = []
            all_images = []
            
            for i, msg_node in enumerate(messages):
                if i >= self.max_messages:
                    logger.warning(f"[Nova-GetChat] 消息数量超过限制 {self.max_messages}")
                    break
                
                sender_name = msg_node.get("sender", {}).get("nickname", "未知用户")
                raw_content = msg_node.get("message") or msg_node.get("content", [])
                
                text_parts = []
                node_images = []
                
                for seg in raw_content:
                    if isinstance(seg, dict):
                        seg_type = seg.get("type")
                        seg_data = seg.get("data", {})
                        
                        if seg_type == "text":
                            text_parts.append(seg_data.get("text", ""))
                        
                        elif seg_type == "image":
                            img_url = self._extract_image_url(seg_data)
                            if img_url:
                                node_images.append(img_url)
                                all_images.append(img_url)
                                
                                # 根据模式处理图片
                                if self.image_mode == "placeholder":
                                    text_parts.append("[图片]")
                                elif self.image_mode == "url":
                                    text_parts.append(f"[图片: {img_url}]")
                                # base64模式在后续处理
                        
                        elif seg_type == "at":
                            qq = seg_data.get("qq", "")
                            text_parts.append(f"[@{qq}]")
                        
                        elif seg_type == "forward":
                            # 嵌套转发
                            if self.enable_nested:
                                nested_id = seg_data.get("id")
                                if nested_id:
                                    text_parts.append("[嵌套转发开始]")
                                    nested_msgs, nested_imgs = await self._extract_forward_content(
                                        event, nested_id, depth + 1
                                    )
                                    for nm in nested_msgs:
                                        text_parts.append(f"  {nm['sender']}: {nm['text']}")
                                    all_images.extend(nested_imgs)
                                    text_parts.append("[嵌套转发结束]")
                            else:
                                text_parts.append("[嵌套转发(已禁用)]")
                
                full_text = "".join(text_parts).strip()
                if full_text:
                    extracted.append({
                        "sender": sender_name,
                        "text": full_text,
                        "images": node_images
                    })
            
            return extracted, all_images
            
        except Exception as e:
            logger.error(f"[Nova-GetChat] 提取合并转发内容失败: {e}")
            logger.error(traceback.format_exc())
            return [], []
    
    def _extract_image_url(self, data: Any) -> Optional[str]:
        """从各种格式中提取图片URL"""
        if not data:
            return None
        
        if isinstance(data, str):
            return data
        
        if isinstance(data, dict):
            # OpenAI格式
            if "image_url" in data:
                url_obj = data["image_url"]
                if isinstance(url_obj, dict):
                    return url_obj.get("url")
            # 简单格式
            if "url" in data:
                return data["url"]
            if "file" in data:
                return data["file"]
        
        if isinstance(data, Image):
            return data.url or data.file
        
        return None
    
    def _format_forward_content(self, messages: List[Dict[str, Any]]) -> str:
        """格式化转发内容为文本"""
        if not messages:
            return ""
        
        # 格式化每条消息
        formatted_msgs = []
        for msg in messages:
            line = self.message_template.format(
                sender=msg["sender"],
                text=msg["text"]
            )
            formatted_msgs.append(line)
        
        # 应用整体模板
        content = "\n".join(formatted_msgs)
        result = self.forward_template.format(
            count=len(messages),
            content=content
        )
        
        return result
    
    async def _generate_summary(self, messages: List[Dict[str, Any]],
                                 event: AstrMessageEvent) -> Optional[str]:
        """使用LLM生成转发内容摘要
        
        返回: 摘要文本，失败时返回None
        """
        if not self.enable_llm_summary:
            return None
        
        # 获取Provider
        provider = None
        if self.summary_provider_id:
            provider = self.context.get_provider_by_id(self.summary_provider_id)
        
        if not provider:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        
        if not provider or not isinstance(provider, Provider):
            logger.warning("[Nova-GetChat] 未找到LLM Provider，跳过摘要生成")
            return None
        
        # 构建摘要内容
        raw_content = ""
        for msg in messages:
            raw_content += f"[{msg['sender']}] {msg['text']}\n"
        
        # 构建prompt
        prompt = self.summary_prompt.format(
            content=raw_content,
            count=len(messages)
        )
        
        try:
            logger.info("[Nova-GetChat] 正在调用LLM生成摘要...")
            
            response = await provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                persist=False
            )
            
            summary = response.completion_text.strip()
            
            if summary:
                logger.info(f"[Nova-GetChat] 摘要生成成功，长度: {len(summary)} 字符")
                return summary
            else:
                logger.warning("[Nova-GetChat] LLM返回空摘要")
                return None
                
        except Exception as e:
            logger.error(f"[Nova-GetChat] LLM摘要生成失败: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def _encode_image_base64(self, url: str) -> Optional[str]:
        """将图片转换为base64"""
        try:
            if url.startswith("base64://"):
                return url.replace("base64://", "data:image/jpeg;base64,")
            
            if url.startswith("http"):
                image_path = await download_image_by_url(url)
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_b64}"
            
            if url.startswith("file:///"):
                path = url.replace("file:///", "")
                with open(path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_b64}"
            
            # 本地路径
            with open(url, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_b64}"
            
        except Exception as e:
            logger.error(f"[Nova-GetChat] 图片base64编码失败: {e}")
            return None
    
    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """处理消息，检测并存储转发内容"""
        if not self.enable_forward_analysis:
            return
        
        # 检测转发消息
        forward_id = await self._detect_forward_message(event)
        if not forward_id:
            return
        
        logger.info(f"[Nova-GetChat] 开始处理转发消息 ID: {forward_id}")
        
        # 提取内容
        messages, images = await self._extract_forward_content(event, forward_id)
        
        if not messages:
            logger.warning("[Nova-GetChat] 未能提取到转发内容")
            return
        
        logger.info(f"[Nova-GetChat] 成功提取 {len(messages)} 条消息, {len(images)} 张图片")
        
        # 尝试生成LLM摘要
        final_text = ""
        is_summary = False
        
        if self.enable_llm_summary:
            summary = await self._generate_summary(messages, event)
            if summary:
                # 使用摘要模板
                final_text = self.summary_inject_template.format(
                    count=len(messages),
                    summary=summary
                )
                is_summary = True
                logger.info("[Nova-GetChat] 使用LLM摘要作为注入内容")
            elif self.keep_original_on_fail:
                # 摘要失败，回退到原始格式
                final_text = self._format_forward_content(messages)
                logger.warning("[Nova-GetChat] 摘要失败，回退到原始转发内容")
            else:
                # 摘要失败且不保留原文，跳过
                logger.warning("[Nova-GetChat] 摘要失败，且未配置保留原文，跳过注入")
                return
        else:
            # 未启用摘要，使用原始格式
            final_text = self._format_forward_content(messages)
        
        # 构建上下文内容
        context_content = []
        context_content.append({"type": "text", "text": final_text})
        
        # 处理图片 (base64模式，仅在非摘要模式或摘要失败时添加图片)
        if self.image_mode == "base64" and images and not is_summary:
            for img_url in images[:10]:  # 限制图片数量
                b64_data = await self._encode_image_base64(img_url)
                if b64_data:
                    context_content.append({
                        "type": "image_url",
                        "image_url": {"url": b64_data}
                    })
        
        # 存储到会话
        self.session_forwards[event.unified_msg_origin].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "content": context_content,
            "raw_messages": messages,
            "images": images,
            "is_summary": is_summary
        })
        
        logger.info(f"[Nova-GetChat] 转发内容已存储到会话 {event.unified_msg_origin}")
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在LLM请求时注入转发内容"""
        session_key = event.unified_msg_origin
        
        if session_key not in self.session_forwards:
            return
        
        forwards = self.session_forwards[session_key]
        if not forwards:
            return
        
        logger.info(f"[Nova-GetChat] 准备注入 {len(forwards)} 条转发记录到上下文")
        
        # 构建注入内容
        inject_content = []
        for fwd in forwards:
            inject_content.extend(fwd["content"])
        
        if not inject_content:
            return
        
        # 根据配置的位置注入
        if self.inject_position == "before_user":
            # 在当前用户消息之前注入
            inject_msg = {
                "role": "user",
                "content": inject_content
            }
            # 找到最后一个用户消息的位置
            insert_pos = len(req.contexts)
            for i in range(len(req.contexts) - 1, -1, -1):
                if req.contexts[i].get("role") == "user":
                    insert_pos = i
                    break
            req.contexts.insert(insert_pos, inject_msg)
            
        elif self.inject_position == "as_system":
            # 作为system消息注入
            text_content = ""
            for item in inject_content:
                if item["type"] == "text":
                    text_content += item["text"]
            
            inject_msg = {
                "role": "system",
                "content": f"用户分享了以下聊天记录，请参考：\n{text_content}"
            }
            req.contexts.append(inject_msg)
            
        elif self.inject_position == "merge_with_user":
            # 合并到用户消息中
            if req.prompt:
                prefix_text = ""
                for item in inject_content:
                    if item["type"] == "text":
                        prefix_text += item["text"] + "\n"
                req.prompt = prefix_text + "\n" + req.prompt
        
        # 清空已处理的转发记录
        self.session_forwards[session_key].clear()
        
        logger.info("[Nova-GetChat] 转发内容已成功注入上下文")
    
    async def terminate(self):
        """插件卸载"""
        logger.info("[Nova-GetChat] 插件已卸载")