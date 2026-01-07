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


@register("nova-getchat", "Nova", "合并转发上下文注入插件 - 自动解析转发消息并注入AI上下文", "1.2.0")
class NovaGetChatPlugin(Star):
    """Nova合并转发上下文注入插件"""
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 会话历史存储
        self.session_forwards = defaultdict(list)
        """存储每个会话的转发消息内容"""
        
        # 摘要缓存：以forward_id为key，避免重复分析同一个转发消息
        self.summary_cache: Dict[str, Dict[str, Any]] = {}
        """格式: {forward_id: {summary, messages, images, timestamp, count}}"""
        
        # 缓存过期时间（秒），默认1小时
        self.cache_ttl = int(self.get_cfg("summary_cache_ttl", 3600))
        
        # 基础配置读取（这些配置不太需要实时更新）
        self.enable_forward_analysis = bool(self.get_cfg("enable_forward_analysis", True))
        self.forward_template = self.get_cfg("forward_template",
            "【转发消息记录 - 共{count}条】\n━━━━━━━━━━━━━━━━━━━━\n{content}\n━━━━━━━━━━━━━━━━━━━━")
        self.message_template = self.get_cfg("message_template", "[{sender}] {text}")
        self.enable_nested = bool(self.get_cfg("enable_nested_forward", True))
        self.max_nested_depth = int(self.get_cfg("max_nested_depth", 3))
        self.max_messages = int(self.get_cfg("max_messages", 50))
        
        logger.info("[Nova-GetChat] 插件已初始化")
        logger.info(f"[Nova-GetChat] 合并转发分析: {'已启用' if self.enable_forward_analysis else '已禁用'}")
        logger.info(f"[Nova-GetChat] 摘要缓存TTL: {self.cache_ttl}秒")
    
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
                                        depth: int = 0, image_mode: str = "placeholder") -> Tuple[List[Dict[str, Any]], List[str]]:
        """提取合并转发消息内容
        
        Args:
            event: 消息事件
            forward_id: 转发消息ID
            depth: 当前嵌套深度
            image_mode: 图片处理模式 (placeholder/url/base64)
        
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
                                
                                # 根据传入的image_mode处理图片
                                if image_mode == "placeholder":
                                    text_parts.append("[图片]")
                                elif image_mode == "url":
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
                                        event, nested_id, depth + 1, image_mode
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
    
    def _is_cache_valid(self, forward_id: str) -> bool:
        """检查缓存是否有效"""
        if forward_id not in self.summary_cache:
            return False
        
        cache_entry = self.summary_cache[forward_id]
        cache_time = datetime.datetime.fromisoformat(cache_entry["timestamp"])
        now = datetime.datetime.now()
        age = (now - cache_time).total_seconds()
        
        return age < self.cache_ttl
    
    def _get_cached_summary(self, forward_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的摘要"""
        if self._is_cache_valid(forward_id):
            logger.info(f"[Nova-GetChat] 命中摘要缓存: {forward_id[:20]}...")
            return self.summary_cache[forward_id]
        return None
    
    def _set_cache(self, forward_id: str, summary: str, messages: List[Dict[str, Any]],
                   images: List[str]):
        """设置摘要缓存"""
        self.summary_cache[forward_id] = {
            "summary": summary,
            "messages": messages,
            "images": images,
            "count": len(messages),
            "timestamp": datetime.datetime.now().isoformat()
        }
        logger.info(f"[Nova-GetChat] 摘要已缓存: {forward_id[:20]}...")
        
        # 清理过期缓存
        self._cleanup_expired_cache()
    
    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        now = datetime.datetime.now()
        expired_keys = []
        
        for fid, entry in self.summary_cache.items():
            cache_time = datetime.datetime.fromisoformat(entry["timestamp"])
            age = (now - cache_time).total_seconds()
            if age >= self.cache_ttl:
                expired_keys.append(fid)
        
        for key in expired_keys:
            del self.summary_cache[key]
        
        if expired_keys:
            logger.debug(f"[Nova-GetChat] 清理了 {len(expired_keys)} 条过期缓存")
    
    async def _generate_summary(self, messages: List[Dict[str, Any]],
                                 event: AstrMessageEvent,
                                 images: List[str] = None) -> Optional[str]:
        """使用LLM生成转发内容摘要
        
        Args:
            messages: 提取的消息列表
            event: 消息事件
            images: 转发中的图片URL列表（可选，用于多模态模型）
        
        返回: 摘要文本，失败时返回None
        """
        # 实时读取配置
        enable_llm_summary = bool(self.get_cfg("enable_llm_summary", False))
        if not enable_llm_summary:
            return None
        
        # 获取Provider - 实时读取provider_id配置
        provider = None
        provider_id = self.get_cfg("summary_provider_id", "")  # 每次调用时重新读取配置
        
        logger.info(f"[Nova-GetChat] 读取到的provider_id: '{provider_id}'")
        
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.warning(f"[Nova-GetChat] 未找到指定Provider: {provider_id}")
        
        # 如果未找到指定provider或未配置，使用当前会话的provider
        if not provider:
            logger.info(f"[Nova-GetChat] 使用会话默认Provider")
            provider = self.context.get_using_provider(event.unified_msg_origin)
        
        if not provider or not isinstance(provider, Provider):
            logger.warning("[Nova-GetChat] 未找到LLM Provider，跳过摘要生成")
            return None
        
        # 构建摘要内容
        raw_content = ""
        for msg in messages:
            raw_content += f"[{msg['sender']}] {msg['text']}\n"
        
        # 实时读取prompt模板
        summary_prompt = self.get_cfg("summary_prompt",
            "请阅读以下聊天记录，并生成一份简洁的摘要。要求：\n1. 保留关键信息和重要观点\n2. 标注主要发言者\n3. 如有讨论结论，请明确指出\n4. 摘要控制在300字以内\n\n聊天记录：\n{content}")
        
        # 构建prompt
        prompt = summary_prompt.format(
            content=raw_content,
            count=len(messages)
        )
        
        # 检查是否需要传图片给多模态模型
        summary_include_images = bool(self.get_cfg("summary_include_images", False))
        summary_max_images = int(self.get_cfg("summary_max_images", 5))
        
        image_urls_to_send = []
        if summary_include_images and images:
            # 限制图片数量，直接传原始URL，让AstrBot的provider自己处理转换
            images_to_process = images[:summary_max_images]
            logger.info(f"[Nova-GetChat] 摘要时传入 {len(images_to_process)} 张图片给多模态模型")
            
            for img_url in images_to_process:
                # 直接使用原始URL，AstrBot的provider会自行转换为base64
                if img_url:
                    image_urls_to_send.append(img_url)
        
        try:
            logger.info("[Nova-GetChat] 正在调用LLM生成摘要...")
            
            # 根据是否有图片选择调用方式
            if image_urls_to_send:
                logger.info(f"[Nova-GetChat] 使用多模态模式，包含 {len(image_urls_to_send)} 张图片")
                response = await provider.text_chat(
                    prompt=prompt,
                    session_id=uuid.uuid4().hex,
                    image_urls=image_urls_to_send,
                    persist=False
                )
            else:
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
        
        # 实时读取配置
        enable_llm_summary = bool(self.get_cfg("enable_llm_summary", False))
        image_mode = self.get_cfg("image_mode", "placeholder")
        keep_original_on_fail = bool(self.get_cfg("keep_original_on_summary_fail", True))
        summary_inject_template = self.get_cfg("summary_inject_template",
            "【转发消息摘要 - 原始{count}条消息】\n{summary}")
        
        # 检查缓存
        cached = self._get_cached_summary(forward_id)
        if cached and enable_llm_summary:
            # 命中缓存，直接使用
            cached_summary = cached.get('summary', '')
            cached_count = cached.get('count', 0)
            
            logger.info(f"[Nova-GetChat] 使用缓存的摘要，原始{cached_count}条消息")
            logger.info(f"[Nova-GetChat] 缓存摘要内容(前100字): {cached_summary[:100] if cached_summary else '(空)'}")
            
            if not cached_summary:
                logger.warning("[Nova-GetChat] 缓存摘要为空！")
                # 缓存无效，继续正常流程重新生成
            else:
                final_text = summary_inject_template.format(
                    count=cached_count,
                    summary=cached_summary
                )
                
                logger.info(f"[Nova-GetChat] 格式化后文本(前100字): {final_text[:100]}")
                
                context_content = [{"type": "text", "text": final_text}]
                
                # 存储到会话
                self.session_forwards[event.unified_msg_origin].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "content": context_content,
                    "raw_messages": cached.get('messages', []),
                    "images": cached.get('images', []),
                    "is_summary": True,
                    "from_cache": True
                })
                
                logger.info(f"[Nova-GetChat] (缓存) 转发内容已存储到会话，content长度: {len(context_content)}")
                return
        
        # 提取内容，传入image_mode参数
        messages, images = await self._extract_forward_content(event, forward_id, depth=0, image_mode=image_mode)
        
        if not messages:
            logger.warning("[Nova-GetChat] 未能提取到转发内容")
            return
        
        logger.info(f"[Nova-GetChat] 成功提取 {len(messages)} 条消息, {len(images)} 张图片")
        
        # 尝试生成LLM摘要
        final_text = ""
        is_summary = False
        
        if enable_llm_summary:
            # 传入images参数，支持多模态摘要
            summary = await self._generate_summary(messages, event, images)
            if summary:
                # 使用摘要模板
                final_text = summary_inject_template.format(
                    count=len(messages),
                    summary=summary
                )
                is_summary = True
                logger.info("[Nova-GetChat] 使用LLM摘要作为注入内容")
                
                # 缓存摘要结果
                self._set_cache(forward_id, summary, messages, images)
                
            elif keep_original_on_fail:
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
        if image_mode == "base64" and images and not is_summary:
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
            "is_summary": is_summary,
            "from_cache": False
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
        
        # 统计缓存命中情况
        cache_hits = sum(1 for f in forwards if f.get("from_cache", False))
        logger.info(f"[Nova-GetChat] 准备注入 {len(forwards)} 条转发记录 (缓存命中: {cache_hits})")
        
        # 构建注入内容 - 提取所有文本
        all_text_parts = []
        all_image_parts = []
        
        for fwd in forwards:
            for item in fwd.get("content", []):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text.strip():
                        all_text_parts.append(text)
                        logger.debug(f"[Nova-GetChat] 提取文本内容: {text[:100]}...")
                elif item.get("type") == "image_url":
                    all_image_parts.append(item)
        
        if not all_text_parts and not all_image_parts:
            logger.warning("[Nova-GetChat] 注入内容为空，跳过注入")
            self.session_forwards[session_key].clear()
            return
        
        # 合并所有文本
        combined_text = "\n".join(all_text_parts)
        logger.info(f"[Nova-GetChat] 合并后文本长度: {len(combined_text)} 字符")
        
        # 实时读取inject_position配置
        inject_position = self.get_cfg("inject_position", "before_user")
        logger.info(f"[Nova-GetChat] 注入位置: {inject_position}")
        
        # 根据配置的位置注入
        if inject_position == "before_user":
            # 添加说明前缀，告诉AI这是用户引用的消息内容
            prefixed_text = f"【以下是用户引用/分享的转发消息内容，请参考后回答用户问题】\n\n{combined_text}"
            
            # 构建content - 如果有图片则使用多模态格式，否则使用纯文本
            if all_image_parts:
                inject_content = [{"type": "text", "text": prefixed_text}]
                inject_content.extend(all_image_parts)
            else:
                # 使用纯文本格式，兼容性更好
                inject_content = prefixed_text
            
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
            logger.info(f"[Nova-GetChat] 在位置 {insert_pos} 注入转发内容")
            
        elif inject_position == "as_system":
            inject_msg = {
                "role": "system",
                "content": f"用户分享了以下聊天记录，请参考：\n{combined_text}"
            }
            req.contexts.append(inject_msg)
            logger.info("[Nova-GetChat] 作为system消息注入")
            
        elif inject_position == "merge_with_user":
            # 合并到用户消息中
            if req.prompt:
                req.prompt = combined_text + "\n\n" + req.prompt
                logger.info("[Nova-GetChat] 合并到用户prompt中")
            else:
                # 如果没有prompt，则添加到contexts最后一个user消息
                for i in range(len(req.contexts) - 1, -1, -1):
                    if req.contexts[i].get("role") == "user":
                        original = req.contexts[i].get("content", "")
                        if isinstance(original, str):
                            req.contexts[i]["content"] = combined_text + "\n\n" + original
                        elif isinstance(original, list):
                            # 多模态格式
                            req.contexts[i]["content"].insert(0, {"type": "text", "text": combined_text + "\n\n"})
                        break
                logger.info("[Nova-GetChat] 合并到最后一个user消息中")
        
        # 清空已处理的转发记录
        self.session_forwards[session_key].clear()
        
        logger.info("[Nova-GetChat] 转发内容已成功注入上下文")
    
    async def terminate(self):
        """插件卸载"""
        logger.info("[Nova-GetChat] 插件已卸载")