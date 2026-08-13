# -*- coding: utf-8 -*-
"""广告识别引擎：本地 RapidOCR / Umi-OCR / 第三方云 API，统一供图片与视频帧识别。

- **local（默认）**：RapidOCR（ONNX），模型不随插件打包——开启 local 引擎时按需
  自动安装 `rapidocr_onnxruntime`（含模型约 30MB）；关闭后**不卸载**，模型保留本地；
- **umi**：Umi-OCR（Rapid 引擎版）HTTP 服务（默认 http://127.0.0.1:1224），需用户
  自行安装运行 Umi-OCR，插件通过 HTTP 调用，专门为视频/图片广告服务；
- **cloud**：第三方云广告检测 API（如阿里云内容安全，通用 JSON 协议），返回是否含广告；
- **llm**（可选，不再默认）：云端视觉模型（智谱 GLM-4V 等），保留兼容；
- **auto**：本地/云优先，识别不到再回退云端视觉。

同步调用放入线程池，不阻塞事件循环；引擎单例懒加载、全局复用。
"""

import asyncio
import base64
import sys

from astrbot.api import logger

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover
    RapidOCR = None


class LocalOCRMixin:
    """广告识别引擎能力，由 ``ModerationMixin`` 组合使用。"""

    def _init_local_ocr(self) -> None:
        """初始化识别引擎：本地引擎槽位、并发锁、可用性标记。"""
        self._local_ocr_engine = None
        self._local_ocr_available = RapidOCR is not None
        self._local_ocr_lock = asyncio.Lock()

    def _ad_engine(self, group_id: str = None) -> str:
        """当前广告识别引擎（local/umi/cloud/llm/auto），非法值回落 local。"""
        try:
            engine = self._cfg_str("ocr_engine", "local", group_id=group_id)
        except Exception:
            engine = "local"
        engine = str(engine or "").strip().lower()
        return engine if engine in ("local", "umi", "cloud", "llm", "auto") else "local"

    # ============================================================
    # 模型按需安装（zip 不带模型，开启 local 引擎时才下载，关闭不卸载）
    # ============================================================

    async def _ensure_local_ocr(self) -> bool:
        """确保本地 RapidOCR 可用：未安装且允许自动安装时 pip 安装（下载模型）。"""
        global RapidOCR
        if RapidOCR is not None:
            return True
        auto = True
        try:
            auto = bool(self.config.get("local_ocr_auto_install", True))
        except Exception:
            auto = True
        if not auto:
            logger.warning("[GroupMgr] 本地OCR未安装且未开启自动安装，本地识别将跳过")
            return False
        logger.info(
            "[GroupMgr] 首次启用本地OCR，自动安装 rapidocr_onnxruntime（含模型约30MB）..."
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", "rapidocr_onnxruntime",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=300)
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: F401
            except ImportError:
                logger.warning("[GroupMgr] 安装后仍未找到 RapidOCR，请手动 pip install rapidocr_onnxruntime")
                return False
            self._local_ocr_available = RapidOCR is not None
            logger.info("[GroupMgr] 本地OCR安装完成（模型已下载，关闭引擎不会卸载）")
            return self._local_ocr_available
        except Exception as exc:
            logger.warning(f"[GroupMgr] 安装本地OCR失败: {exc}")
            return False

    # ============================================================
    # 本地 RapidOCR
    # ============================================================

    async def _local_ocr_text(self, data: bytes) -> str:
        """对图片字节执行本地 RapidOCR，返回识别文字；失败/不可用返回空串。"""
        if not await self._ensure_local_ocr():
            return ""
        if not data:
            return ""
        if self._local_ocr_engine is None:
            async with self._local_ocr_lock:
                if self._local_ocr_engine is None:
                    try:
                        self._local_ocr_engine = RapidOCR(
                            det_use_cuda=False,
                            rec_use_cuda=False,
                            use_cls=False,
                            show_log=False,
                        )
                    except Exception as exc:
                        logger.warning(f"[GroupMgr] 初始化 RapidOCR 失败: {exc}")
                        self._local_ocr_engine = False
        if not self._local_ocr_engine:
            return ""
        try:
            result, _ = await asyncio.to_thread(self._local_ocr_engine, data)
            if not result:
                return ""
            texts = []
            for line in result:
                if len(line) > 1 and line[1]:
                    texts.append(str(line[1]))
            return " ".join(texts).strip()
        except Exception as exc:
            logger.debug(f"[GroupMgr] 本地 OCR 识别失败: {exc}")
            return ""

    # ============================================================
    # Umi-OCR（Rapid 引擎版，本地 HTTP 服务）
    # ============================================================

    async def _umi_ocr_text(self, data: bytes) -> str:
        """调用 Umi-OCR 的 HTTP API 识别图片文字（默认 http://127.0.0.1:1224）。"""
        if not data:
            return ""
        url = self._cfg_str("umi_ocr_url", "http://127.0.0.1:1224").strip().rstrip("/")
        if not url:
            return ""
        try:
            import aiohttp

            form = aiohttp.FormData()
            form.add_field("image", data, filename="image.jpg", content_type="image/jpeg")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/api/ocr",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return ""
                    result = await resp.json()
            data_obj = result.get("data") or {}
            texts = data_obj.get("texts") or []
            lines = []
            for item in texts:
                if isinstance(item, dict):
                    t = item.get("text")
                    if t:
                        lines.append(str(t))
                elif isinstance(item, str):
                    lines.append(item)
            return " ".join(lines).strip()
        except Exception as exc:
            logger.debug(f"[GroupMgr] Umi-OCR 调用失败: {exc}")
            return ""

    # ============================================================
    # 第三方云广告检测 API（如阿里云内容安全）
    # ============================================================

    async def _cloud_audit_image(self, data: bytes) -> tuple:
        """调用第三方云广告检测 API（通用 JSON 协议）。返回 (is_ad, reason)。

        对接格式（需第三方服务支持）：
        POST {cloud_audit_url}  请求头 Authorization: Bearer {cloud_audit_api_key}
        {"image_base64": "<base64>"}  →  {"is_ad": bool, "score": float, "reason": str}
        """
        if not data:
            return False, ""
        url = self._cfg_str("cloud_audit_url", "").strip()
        if not url:
            logger.debug("[GroupMgr] 未配置 cloud_audit_url，云广告检测不可用")
            return False, ""
        api_key = self._cfg_str("cloud_audit_api_key", "").strip()
        threshold = 0.8
        try:
            threshold = float(self.config.get("cloud_audit_threshold", 0.8) or 0.8)
        except (TypeError, ValueError):
            threshold = 0.8
        try:
            import aiohttp

            payload = {"image_base64": base64.b64encode(data).decode("ascii")}
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return False, ""
                    result = await resp.json()
            is_ad = bool(result.get("is_ad", False))
            score = float(result.get("score", 0.0) or 0.0)
            reason = str(result.get("reason", "") or "")
            if is_ad or score >= threshold:
                return True, reason or f"云API广告检测(score={score:.2f})"
            return False, ""
        except Exception as exc:
            logger.debug(f"[GroupMgr] 云广告检测调用失败: {exc}")
            return False, ""

    # ============================================================
    # 统一识别入口（非 LLM 引擎）
    # ============================================================

    async def _detect_media_text(self, data: bytes, group_id: str = None) -> str:
        """按当前引擎识别图片/视频帧中的文字（或云 API 广告判定）。

        返回识别文本；云 API 命中广告时返回含 [云API] 标记的文本。
        """
        engine = self._ad_engine(group_id)
        if engine == "umi":
            return await self._umi_ocr_text(data)
        if engine == "cloud":
            is_ad, reason = await self._cloud_audit_image(data)
            if is_ad:
                return f"[云API] 广告：{reason}"
            return ""
        # local 或 auto 的本地部分
        return await self._local_ocr_text(data)

    def _engine_prefer_llm(self, group_id: str = None) -> bool:
        """当前引擎是否需要 LLM 视觉（llm / auto 的回退路径）。"""
        return self._ad_engine(group_id) in ("llm", "auto")

    def _engine_cloud_only(self, group_id: str = None) -> bool:
        """当前引擎是否纯云 API（无本地 OCR 文字）。"""
        return self._ad_engine(group_id) == "cloud"
