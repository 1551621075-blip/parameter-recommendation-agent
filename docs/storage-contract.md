# 本地持久化接口（原创工程）

`storage.py` 单独实现通用 SQLite/制品原语，不依赖 domain.py 之外的业务。使用 domain.canonical/digest 或模块自有等价严格 JSON。所有 public payload 必须 JSON，禁止 pickle。数据库 `root/data.sqlite3`，制品 `root/artifacts/<sha256>.json`。

## Principal 与 Store

- frozen dataclass `Principal(user_id: str, role: str, plants: tuple[str,...], acl_version: int)`；角色 admin/reviewer/operator。
- `Store(root: Path)`：初始化/打开，只接受安全绝对解析目录、普通单链接文件，不跟 symlink；新目录权限0700。不得打开其他工程。
- `store.transaction()` contextmanager -> sqlite3.Connection：BEGIN IMMEDIATE，row_factory sqlite3.Row，WAL/foreign_keys/trusted_schema=OFF；失败rollback，不跨 provider call 持有事务。
- `store.connect()` contextmanager -> readonly-style业务读连接（查询用，允许初始化PRAGMA）；安全检查与事务一致。
- `store.create_user(user_id, token, role, plants: list[str]) -> Principal`：严格验证，token随机由CLI生成，存 SHA256 不存明文。已存在拒绝。不在审计/错误/异常带token。
- `store.authenticate(token: str) -> Principal`：过长/非法/未知/停用统一unauthorized401；不持有token。
- `store.require(principal, plant_id=None, roles=None, *, conn=None) -> None`：每次重验当前用户active、role/plants/acl_version及可选工厂/角色，撤权后旧Principal不得用；传conn时同一事务。
- `store.set_user(user_id, *, active=None, role=None, plants=None) -> None`：仅本地管理原语，变更acl_version，不开放匿名API。

## 制品与对象

- `store.put_artifact(payload: dict) -> str`：严格有界JSON(<=2MiB,finite,depth<=40)，SHA256、原子无覆盖发布、fsync、单链接无symlink；相同hash已有文件也重验。
- `store.get_artifact(content_hash: str) -> dict`：完整hash/类型/路径/schema边界检查；缺失或损坏 `DomainError('integrity',...,409)`。
- `store.create_object(conn, *, kind, object_id, principal, plant_id, status, payload) -> dict`：same-tx require；owner=principal.user_id。version=1；重复409。返回统一对象。
- `store.get_object(kind, object_id, principal, *, conn=None) -> dict`：require+owner/plant；未授权/不存在不泄露，404；重验制品hash。返回 `{'kind','object_id','owner','plant_id','status','version','content_hash','payload'}`。
- `store.update_object(conn, *, kind, object_id, principal, expected_version, status, payload) -> dict`：same-tx require/owner/version CAS，version+1。
- `store.list_objects(kind, principal, *, limit=100) -> list[dict]`：按owner+当前plant授权过滤、有界<=100；返回统一对象列表。
- `store.audit(conn, principal, *, kind, object_id, action, detail:dict) -> None`：内容以artifact保存，行不含rawtoken；对象应存在且所属授权。
- `store.audit_events(principal, *, limit=100) -> list[dict]`：仅本人/当前plants，有界；包含 kind,object_id,action,detail,created_at_ms。

## 幂等回执

- `store.receipt(conn, principal, action_id:str, request:dict) -> dict|None`：主键(user_id,action_id)，same内容返回原response artifact；不同内容409 `idempotency_conflict`。request hash以operation+target+payload完整绑定，由业务传入。必须重新require身份；目标授权仍由业务get_object先检查。
- `store.record_receipt(conn, principal, action_id, request, response:dict) -> None`：同事务，重复同内容/同response幂等；不同拒绝。

## 可查询表（agent 独占 calls）

`calls` 字段：call_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, owner TEXT NOT NULL, lease_id TEXT NOT NULL, status TEXT NOT NULL CHECK status IN ('started','completed','failed','unknown'), tokens INTEGER NULL CHECK tokens>=0, elapsed_s REAL NOT NULL DEFAULT0, response_hash TEXT NULL, error_code TEXT NULL, created_at_ms INTEGER NOT NULL。

Agent 可以在transaction中直接SQL这些列，不得执行模型SQL。对象均用Store API，字段object状态存在artifact中，禁止把多表有相互依赖的变更拆开提交。

Store 不自动创建默认用户/模型/演示/批准；不访问模型、网络、用户profile。所有报错message为安全通用说明。仅本机文件，不保证分布式 exactly-once。
