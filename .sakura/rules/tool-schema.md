# AstrBot 工具 Schema 规则

适用于 `llm_tools.py` 等暴露给模型提供商的函数工具。

## 1. Schema 是独立接口层

必须区分：

1. 文档/装饰器或 Python 注解中的工具声明；
2. AstrBot 生成的最终 JSON Schema；
3. `_normalize_id_list` 等业务执行层。

执行函数能接收列表，不代表声明合法。遇到 `function_declarations`、`properties`、`items` 等 400 错误，应先检查最终 Schema，再检查声明来源，最后才检查业务代码。

## 2. 数组参数约束

每个数组节点都必须带合法元素 Schema：

```json
{
  "type": "array",
  "items": { "type": "string" }
}
```

AstrBot 文档参数可优先验证 `list[string]` 语法。修改 Python 注解前必须确认目标 AstrBot 版本：

- Schema 来源是文档、注解还是两者合并；
- 两者冲突时谁优先；
- 是否支持 `list[str]`；
- 是否稳定支持 `Union` 或 `str | list[str]`。

未经验证不要为“类型一致”贸然引入联合类型。

## 3. 提供商兼容性

Gemini 会在执行前严格校验**全部**函数声明；一个非法工具即使未被调用，也可能阻断整次请求。定级前须确认：

- 问题工具是否默认注册；
- 是否使全部 Gemini 请求失败；
- 是否仅影响严格校验的提供商；
- 是否有禁用工具或切换提供商的临时方案。

默认注册且阻断全部请求可定 `high`；仅显式启用时通常为 `medium`。

## 4. 测试要求

- 从实际生成结果断言最终 JSON Schema，而非只测业务函数。
- 按**工具名称**定位声明，禁止依赖 `function_declarations[23]` 等易变索引。
- 增加通用递归断言：所有 `type: array` 节点均存在有效 `items`。
- 对批量禁言、批量踢人等具体工具断言 `items.type == string`。
- 外部 Gemini 调用只作为可选冒烟；CI 主保障必须是本地 Schema 测试。
- 修复后回归其他工具和提供商，避免声明生成器变化引入连锁问题。
