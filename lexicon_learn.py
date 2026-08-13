# -*- coding: utf-8 -*-
"""自适应上下文学习模块（v2.8.0）。

让插件从群聊上下文中自动学习"本群特有"的引流/广告黑话（如「AI中转」「plus日抛team」），
把候选词沉淀进按群独立的自训练词库（learned_keywords 表），逐步自我增强审核能力。

安全设计（自动加违禁词是高危操作，误加常用词会误删满群消息，故以安全为第一原则）：
- **默认候选/审核模式**：AI 只产出"建议词"，管理员在 WebUI「词库学习」页审批后才生效；
  auto_apply（自动生效）默认关闭，且有置信度 + 出现次数双阈值。
- **按群独立**：A 群学到的词只在 A 群匹配，不污染其它群。
- **多重防误杀**：最小词长、跨轮出现次数阈值、白名单保护、置信度阈值、单轮上限。
- 全部能力受 lexicon_learn_enabled（默认关）+ enabled + disclaimer_agreed 三重门控。

工作流：
  被动采集（_learn_observe_message，消息管线每条 append 进按群环形缓冲）
    → 周期挖掘（scheduler 的 _lexicon_learn_loop → _run_lexicon_learning，默认 300s）
      → LLM 从缓冲样本里找出不在已知词库、反复出现的引流词，带置信度/理由/样例
        → 过滤 + 累计出现次数 → 存为 pending 候选（或高置信自动生效）
          → 管理员审批 → approved → 并入该群的学习匹配器 → 参与初筛
"""
import json
import re
import time
from collections import deque

from astrbot.api import logger

try:
    from .automaton import HybridMatcher
except ImportError:  # 独立加载（单元测试用 spec_from_file_location）时无包上下文
    from automaton import HybridMatcher

# 单群消息缓冲上限（条）；超出后按 FIFO 丢弃最旧，防止内存无界增长。
# 每轮挖掘取最近 sample_size 条后清空缓冲，故上限贴近样本硬上限即可，
# 设过大只会占内存而不会被用到（每轮至多消费 _LEARN_SAMPLE_HARD_CAP 条）。
_LEARN_BUFFER_MAX = 300
# 最多跟踪多少个群的缓冲，超过则不再为新群建缓冲（极端保护，正常远不会触及）。
_LEARN_MAX_GROUPS = 1000
# 单条消息进缓冲时的截断长度。
_LEARN_MSG_MAX_CHARS = 200
# 至少积累这么多条消息才触发一次挖掘（样本太少 LLM 判断不可靠且浪费调用）。
_LEARN_MIN_MESSAGES = 15
# 挖掘时喂给 LLM 的单批样本上限（再受 lexicon_learn_sample_size 配置约束）。
_LEARN_SAMPLE_HARD_CAP = 300
# 候选关键词硬上限长度，防止 LLM 返回整句当"关键词"。
_LEARN_KEYWORD_MAX_CHARS = 40
# 单轮单群最多接纳的候选数硬上限（再受 lexicon_learn_max_per_run 配置约束）。
_LEARN_MAX_PER_RUN_HARD_CAP = 50

# 内置永不学习的通用词（避免把常见词/协议片段当违禁词）；用户可再配 lexicon_learn_allowlist 扩充。
_LEARN_BUILTIN_ALLOWLIST = frozenset({
    "http", "https", "www", "com", "cn", "net", "org", "qq", "the", "and", "you",
    "群", "我", "你", "他", "的", "了", "吗", "呢", "啊", "哈", "哈哈", "谢谢", "大佬",
    "机器人", "管理", "群主", "消息", "在吗", "早上好", "晚安", "签到",
})


class LexiconLearnMixin:
    # ---------------- 初始化 / 状态 ----------------
    def _init_lexicon_learn(self) -> None:
        """在 Main.__init__ 中调用。初始化内存状态（不含后台任务，任务由 scheduler 起）。"""
        self._learn_buffers = {}            # group_id -> deque[str]
        self._learn_matchers = {}           # group_id -> {"ad": HybridMatcher, "swear": HybridMatcher}
        self._learn_task = None             # scheduler 持有的后台任务句柄

    def _load_learned_matchers(self) -> None:
        """启动时从 DB 载入全部已审批学习词，构建按群匹配器。"""
        try:
            approved = self._storage.list_approved_learned("")
        except Exception as e:
            logger.debug(f"[GroupMgr] 载入学习词失败: {e}")
            return
        by_group = {}
        for row in approved:
            by_group.setdefault(str(row["group_id"]), []).append(row)
        self._learn_matchers = {}
        for gid, rows in by_group.items():
            self._build_learn_matcher_from_rows(gid, rows)
        if self._learn_matchers:
            total = sum(len(r) for r in by_group.values())
            logger.info(f"[GroupMgr] 已载入 {total} 条学习词（{len(self._learn_matchers)} 个群）")

    def _build_learn_matcher_from_rows(self, group_id: str, rows: list) -> None:
        cats = {"ad": [], "swear": []}
        for row in rows:
            cat = row.get("category") if isinstance(row, dict) else None
            kw = row.get("keyword") if isinstance(row, dict) else None
            if not kw:
                continue
            cats["swear" if cat == "swear" else "ad"].append(kw)
        built = {}
        for cat, kws in cats.items():
            if not kws:
                continue
            m = HybridMatcher()
            m.add_literal_keywords(kws)
            m.build()
            built[cat] = m
        if built:
            self._learn_matchers[str(group_id)] = built
        else:
            self._learn_matchers.pop(str(group_id), None)

    def _rebuild_learn_matcher(self, group_id: str) -> None:
        """某群学习词有增删（审批/拒绝/删除）后重建该群匹配器。"""
        try:
            rows = self._storage.list_approved_learned(str(group_id))
        except Exception as e:
            logger.debug(f"[GroupMgr] 重建学习匹配器失败({group_id}): {e}")
            return
        self._build_learn_matcher_from_rows(str(group_id), rows)

    # ---------------- 被动采集（消息管线调用） ----------------
    def _learn_observe_message(self, group_id: str, text: str) -> None:
        """把一条群消息文本纳入学习缓冲。极轻量，热路径调用，任何异常都吞掉不影响审核。"""
        try:
            if not group_id or not text:
                return
            if not self._cfg("lexicon_learn_enabled", False, group_id=str(group_id)):
                return
            gid = str(group_id)
            buf = self._learn_buffers.get(gid)
            if buf is None:
                if len(self._learn_buffers) >= _LEARN_MAX_GROUPS:
                    return
                buf = deque(maxlen=_LEARN_BUFFER_MAX)
                self._learn_buffers[gid] = buf
            snippet = text.strip()[:_LEARN_MSG_MAX_CHARS]
            if snippet:
                buf.append(snippet)
        except Exception:
            pass

    # ---------------- 匹配（初筛调用） ----------------
    def _learned_hit(self, group_id: str, text: str):
        """返回命中的类别（'swear'/'ad'）或 None。swear 优先（更严重）。"""
        if not group_id or not text:
            return None
        matchers = self._learn_matchers.get(str(group_id))
        if not matchers:
            return None
        try:
            for cat in ("swear", "ad"):
                m = matchers.get(cat)
                if m is not None and m.is_match(text):
                    return cat
        except Exception:
            return None
        return None

    # ---------------- 审批 / 生效（web 层调用） ----------------
    def _learn_set_status(self, keyword_id: int, status: str) -> dict:
        """审批/拒绝一条候选并按需重建匹配器。返回被更新行信息（或 {}）。"""
        row = self._storage.set_learned_status(int(keyword_id), status, int(time.time()))
        if row and row.get("group_id"):
            self._rebuild_learn_matcher(row["group_id"])
        return row or {}

    def _learn_delete(self, keyword_id: int) -> dict:
        row = self._storage.delete_learned(int(keyword_id))
        if row and row.get("group_id") and row.get("status") == "approved":
            self._rebuild_learn_matcher(row["group_id"])
        return row or {}

    # ---------------- 挖掘主流程（scheduler 调用） ----------------
    def _learn_group_active(self, group_id: str) -> bool:
        gid = str(group_id)
        return bool(
            self.config.get("disclaimer_agreed", False)
            and self._cfg("enabled", True, group_id=gid)
            and self._cfg("lexicon_learn_enabled", False, group_id=gid)
        )

    def _lexicon_learn_any_group_enabled(self) -> bool:
        """scheduler 门控：是否有任何群开启了上下文学习（含缓冲已有数据的群）。"""
        if not self.config.get("disclaimer_agreed", False):
            return False
        if self._cfg("enabled", True) and self._cfg("lexicon_learn_enabled", False):
            return True
        for gid in list(self._learn_buffers.keys()):
            if self._learn_group_active(gid):
                return True
        group_ids = set()
        for attr in ("_group_white_set",):
            group_ids.update(str(x) for x in (getattr(self, attr, set()) or set()) if x)
        try:
            group_ids.update(str(x) for x in self._storage.list_configured_groups() if x)
        except Exception:
            pass
        return any(self._learn_group_active(gid) for gid in group_ids)

    async def _run_lexicon_learning(self) -> None:
        """扫描所有有缓冲数据且启用的群，逐群挖掘候选词。"""
        for gid in list(self._learn_buffers.keys()):
            if not self._learn_group_active(gid):
                continue
            try:
                await self._learn_mine_group(gid)
            except Exception as e:
                logger.warning(f"[GroupMgr] 群 {gid} 上下文学习出错: {e}")

    async def _learn_mine_group(self, group_id: str) -> None:
        gid = str(group_id)
        buf = self._learn_buffers.get(gid)
        if not buf or len(buf) < _LEARN_MIN_MESSAGES:
            return
        sample_size = self._cfg_int("lexicon_learn_sample_size", 60, group_id=gid)
        sample_size = max(_LEARN_MIN_MESSAGES, min(sample_size, _LEARN_SAMPLE_HARD_CAP))
        messages = list(buf)[-sample_size:]
        buf.clear()  # 已消费的样本清空，避免重复挖掘同一批消息

        known = [r["keyword"] for r in self._storage.list_learned(gid, "", limit=500, offset=0)]
        raw = await self._learn_call_llm(gid, messages, known)
        if not raw:
            return
        candidates = self._learn_parse_candidates(raw)
        if not candidates:
            return

        min_len = self._cfg_int("lexicon_learn_min_length", 2, group_id=gid)
        min_conf = self._learn_cfg_float("lexicon_learn_min_confidence", 0.75, gid)
        min_occ = self._cfg_int("lexicon_learn_min_occurrences", 3, group_id=gid)
        max_per_run = min(
            self._cfg_int("lexicon_learn_max_per_run", 10, group_id=gid),
            _LEARN_MAX_PER_RUN_HARD_CAP,
        )
        auto_apply = self._cfg("lexicon_learn_auto_apply", False, group_id=gid)
        allowlist = self._learn_allowlist(gid)

        now = int(time.time())
        accepted = 0
        applied = 0
        seen_this_round = set()  # 单轮内去重：LLM 可能重复返回同词，避免一轮内把出现次数刷高
        for cand in candidates:
            if accepted >= max_per_run:
                break
            kw = cand.get("keyword", "")
            if not self._learn_keyword_ok(kw, min_len, allowlist):
                continue
            kw_norm = kw.strip().lower()
            if kw_norm in seen_this_round:
                continue
            seen_this_round.add(kw_norm)
            category = cand.get("category", "ad")
            category = "swear" if category == "swear" else "ad"
            confidence = cand.get("confidence", 0.0)
            reason = str(cand.get("reason", ""))[:200]
            sample = str(cand.get("sample", ""))[:200]
            try:
                info = self._storage.upsert_learned_candidate(
                    gid, kw, category, reason, sample, confidence, now, source="llm")
            except Exception as e:
                logger.debug(f"[GroupMgr] 写学习候选失败({gid}/{kw}): {e}")
                continue
            accepted += 1
            # 自动生效：仅在 auto_apply 开启且候选仍 pending 时，走【多维度自动审批门】。
            # 必须同时通过 置信度/出现频次/形态/非常用词/对抗式LLM复核 各维度才自动生效，
            # 否则保持 pending 转人工，避免误加常用词导致满群误撤回。
            if auto_apply and info.get("status") == "pending":
                verdict = await self._learn_multidim_approval(
                    gid, kw, category, confidence, sample,
                    info.get("occurrences", 1), min_conf, min_occ, min_len)
                if verdict.get("approved"):
                    self._learn_approve_by_keyword(gid, kw)
                    applied += 1
                    logger.info(f"[GroupMgr] 群 {gid} 学习词自动审批通过「{kw}」：{verdict.get('note','')}")
        if accepted:
            logger.info(
                f"[GroupMgr] 群 {gid} 上下文学习：新增/累计 {accepted} 个候选"
                + (f"，自动生效 {applied} 个" if applied else "，待审批"))

    def _learn_approve_by_keyword(self, group_id: str, keyword: str) -> None:
        row = self._storage.get_learned_keyword(str(group_id), str(keyword))
        if row and row.get("id"):
            self._learn_set_status(int(row["id"]), "approved")

    # ---------------- 多维度自动审批门 ----------------
    async def _learn_multidim_approval(self, group_id: str, keyword: str, category: str,
                                       confidence: float, sample: str, occurrences: int,
                                       min_conf: float, min_occ: int, min_len: int) -> dict:
        """多维度自动审批：候选词要自动生效（进而自动参与撤回判断），必须【全部维度通过】。

        维度：① 置信度达标 ② 跨轮出现频次达标 ③ 词形合理 ④ 非白名单常用词
              ⑤ 对抗式 LLM 复核（专门反驳"是否会误伤正常消息"，可关）。
        任一维度不过即保持 pending 转人工，宁可漏批不可误封。
        """
        dims = []

        def add(name, ok, detail=""):
            dims.append({"name": name, "pass": bool(ok), "detail": detail})

        add("置信度", confidence >= min_conf, f"{confidence:.2f}≥{min_conf}")
        add("出现频次", occurrences >= min_occ, f"{occurrences}≥{min_occ}")
        add("词形", len(keyword) >= max(2, int(min_len)) and not keyword.isdigit(),
            f"长度{len(keyword)}")
        add("非常用词", keyword.strip().lower() not in self._learn_allowlist(group_id))

        # 前置维度全过才进对抗复核（省调用）；对抗复核默认开启。
        base_pass = all(d["pass"] for d in dims)
        if base_pass and self._cfg("lexicon_learn_verify_llm", True, group_id=str(group_id)):
            safe = await self._learn_verify_safe_to_ban(group_id, keyword, category, sample)
            add("对抗复核", safe,
                "可安全自动拦截" if safe else "疑似常用词/可能误伤")

        approved = all(d["pass"] for d in dims)
        note = "；".join(f"{d['name']}{'✓' if d['pass'] else '✗'}" for d in dims)
        return {"approved": approved, "dims": dims, "note": note}

    async def _learn_verify_safe_to_ban(self, group_id: str, keyword: str,
                                        category: str, sample: str) -> bool:
        """对抗式复核：让模型【尝试反驳】把该词设为违禁词的安全性，从严默认不安全。"""
        system_prompt = (
            "你是严格的审核安全员。有人想把某个词设为群违禁词（命中即可能自动撤回消息）。"
            "你的职责是【尽力反驳】，找出它会误伤正常聊天的风险。只返回严格 JSON。"
        )
        cat_cn = "广告引流" if category != "swear" else "辱骂"
        prompt = (
            f"拟自动把关键词【{keyword}】加入本群违禁词（{cat_cn}类），"
            f"命中该词的消息可能被自动撤回甚至禁言。触发样例：{sample or '（无）'}\n"
            "请判断：把它设为违禁词是否会误伤正常聊天？它是不是常见词、正常商品名、"
            "普通口语、人名地名或有其它正常含义？\n"
            "判定从严：只要存在一点误伤正常消息的可能，就判定不安全。\n"
            '严格返回：{"safe_to_ban": true/false, "reason": "简短理由"}'
        )
        provider_id = self._cfg_str("lexicon_learn_provider_id", "", group_id=str(group_id)).strip()
        try:
            if provider_id:
                from .moderation import _LLMErrorBag
                errors = _LLMErrorBag()
                raw = await self._run_llm_with_limits(
                    lambda: self._call_llm_by_provider_id(provider_id, system_prompt, prompt, errors))
            else:
                raw = await self._run_llm_with_limits(
                    lambda: self._call_llm_safe(system_prompt, prompt))
            m = re.search(r"\{.*\}", str(raw), re.DOTALL)
            if not m:
                return False
            data = json.loads(m.group())
            return bool(data.get("safe_to_ban", False))
        except Exception as e:
            logger.debug(f"[GroupMgr] 学习词对抗复核失败({keyword})，保守转人工: {e}")
            return False

    # ---------------- LLM 调用与解析 ----------------
    async def _learn_call_llm(self, group_id: str, messages: list, known: list) -> str:
        system_prompt = (
            "你是群管审核助手，负责从群聊记录里发现【本群特有的引流/广告/售卖黑话】。"
            "只返回严格 JSON，不要任何解释性文字。"
        )
        known_text = "、".join(known[:100]) if known else "（暂无）"
        sample_text = "\n".join(f"- {m}" for m in messages[:_LEARN_SAMPLE_HARD_CAP])
        prompt = (
            "下面是某 QQ 群的近期聊天记录。请找出【反复出现、明显是广告/引流/售卖/拉人】"
            "但可能还没被现有词库拦截的关键词或短语，例如卖 AI 中转账号、卖隐形眼镜"
            "（如 plus日抛/team）、拉人进群、私下交易等。\n\n"
            "严格要求：\n"
            "1. 只挑【真正的广告/引流黑话或商品/服务名】，绝不要挑正常聊天用词、常见口语、"
            "人名、地名、普通商品名。宁可少挑也不要误伤正常词。\n"
            "2. 关键词要具体且有辨识度（如「AI中转」「日抛plus」「加V」），不要给太宽泛的单字或常用词。\n"
            "3. 跳过这些已知词，不要重复：" + known_text + "\n"
            "4. 每个词给出 category（ad=广告引流 / swear=辱骂）、confidence（0~1 置信度）、"
            "reason（简短理由）、sample（触发的原文片段）。\n"
            "5. 最多返回 10 个，没有就返回空数组。\n\n"
            "【聊天记录】\n" + sample_text + "\n\n"
            '严格返回 JSON 数组，形如：'
            '[{"keyword":"AI中转","category":"ad","confidence":0.9,"reason":"多次售卖AI中转账号","sample":"低价出AI中转"}]'
        )
        provider_id = self._cfg_str("lexicon_learn_provider_id", "", group_id=str(group_id)).strip()
        try:
            if provider_id:
                from .moderation import _LLMErrorBag
                errors = _LLMErrorBag()
                return await self._run_llm_with_limits(
                    lambda: self._call_llm_by_provider_id(provider_id, system_prompt, prompt, errors))
            return await self._run_llm_with_limits(
                lambda: self._call_llm_safe(system_prompt, prompt))
        except Exception as e:
            logger.debug(f"[GroupMgr] 学习 LLM 调用失败({group_id}): {e}")
            return ""

    @staticmethod
    def _learn_parse_candidates(raw: str) -> list:
        if not raw:
            return []
        text = raw.strip()
        # 容错：从返回里抠出 JSON 数组
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            text = m.group()
        try:
            data = json.loads(text)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            kw = str(item.get("keyword", "")).strip()
            if not kw:
                continue
            try:
                conf = float(item.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            out.append({
                "keyword": kw,
                "category": str(item.get("category", "ad")).strip().lower(),
                "confidence": conf,
                "reason": str(item.get("reason", "")),
                "sample": str(item.get("sample", "")),
            })
        return out

    # ---------------- 过滤与配置助手 ----------------
    def _learn_allowlist(self, group_id: str) -> frozenset:
        extra = self.config.get("lexicon_learn_allowlist", []) or []
        if not isinstance(extra, list):
            extra = [extra]
        extra_norm = {str(x).strip().lower() for x in extra if str(x).strip()}
        return frozenset(_LEARN_BUILTIN_ALLOWLIST | extra_norm)

    @staticmethod
    def _learn_keyword_ok(keyword: str, min_len: int, allowlist: frozenset) -> bool:
        kw = str(keyword).strip()
        if not kw:
            return False
        if len(kw) > _LEARN_KEYWORD_MAX_CHARS:
            return False
        if len(kw) < max(2, int(min_len)):
            return False
        low = kw.lower()
        if low in allowlist:
            return False
        # 纯数字/纯标点不作为关键词（易误伤）
        if kw.isdigit():
            return False
        if not re.search(r"[0-9A-Za-z一-鿿]", kw):
            return False
        return True

    def _learn_cfg_float(self, key: str, default: float, group_id: str) -> float:
        """读取 float 配置（群覆盖优先），带范围钳制到 [0,1]。"""
        try:
            gv = self._get_group_override(group_id, key)
            if gv is not None:
                val = float(gv)
            else:
                meta = self._config_schema.get(key, {}) if hasattr(self, "_config_schema") else {}
                raw = self.config.get(key, meta.get("default", default))
                val = float(raw)
        except (TypeError, ValueError):
            val = default
        return max(0.0, min(1.0, val))
