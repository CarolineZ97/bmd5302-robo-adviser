# 部署指引 · BMD5302 Robo-Adviser → Streamlit Community Cloud

> 目标：把本仓库变成一个公网可访问的免费网站，形如
> `https://bmd5302-robo-adviser-caroline.streamlit.app`

## 前置已就绪 ✅

- [x] 独立仓库目录 `deploy/bmd5302-robo-adviser/`
- [x] `app.py` 已在根，入口合规
- [x] `.streamlit/config.toml` 主题配置
- [x] `.streamlit/secrets.toml.example` 模板
- [x] `packages.txt`（apt 依赖，为 PDF 字体）
- [x] `runtime.txt` 锁定 Python 3.12
- [x] `requirements.txt` 已 pin 版本 + 加 kaleido
- [x] `config.py` 同时支持 `.env` 和 `st.secrets`
- [x] `.gitignore` 过滤 `.env`、`secrets.toml`、PDF 导出目录
- [x] Git 本地 commit `7d98073`

## 你需要执行的 3 步

### Step 1 · 登录 GitHub CLI（1 分钟）

打开 **PowerShell**，运行：

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" auth login
```

按提示选择：
- `GitHub.com`
- `HTTPS`
- `Yes`（用 git 操作也认证）
- `Login with a web browser`
- 记下 8 位一次性码 → 回车 → 浏览器自动打开 → 粘贴码 → 授权

完成后：

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" auth status
```

应看到 `Logged in to github.com account CarolineZ97`。

### Step 2 · 创建公开仓库并推送（30 秒）

```powershell
cd "C:\Caroline\BMD 5302\deploy\bmd5302-robo-adviser"
& "C:\Program Files\GitHub CLI\gh.exe" repo create bmd5302-robo-adviser --public --source=. --remote=origin --push
```

成功后会打印仓库 URL，类似：
`https://github.com/CarolineZ97/bmd5302-robo-adviser`

### Step 3 · 在 Streamlit Cloud 点一下（2 分钟）

1. 打开 <https://share.streamlit.io>
2. 右上角 **Sign in with GitHub** → 授权
3. **Create app** → **Deploy a public app from GitHub**
4. 填：
   - **Repository:** `CarolineZ97/bmd5302-robo-adviser`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** 改成你喜欢的（默认会自动生成）
5. 展开 **Advanced settings** → **Secrets**，粘贴下面内容（**替换** `sk-xxx` 为真实 DeepSeek key）：

   ```toml
   OPENAI_API_KEY = "sk-你的真实key"
   OPENAI_BASE_URL = "https://api.deepseek.com/v1"
   LLM_MODEL = "deepseek-chat"
   ```

   > 没 key 也可以跳过——应用会进入 Mock 模式。
6. 点 **Deploy**。首次构建约 3 分钟（装依赖），之后打开非常快。

## 部署后

- 应用地址复制到本仓库 `README.md` 第一行的 `Live demo:` 占位处。
- 在 Part 4 视频里直接演示公网地址（老师任何地方都能打开）。
- 每次 `git push` 到 `main` → Streamlit Cloud 自动重建，无需手动操作。

## 常见报错速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError` | 忘了加依赖 | 在 `requirements.txt` 补上，push 触发重建 |
| 构建卡在 `Installing dependencies` 5 分钟+ | numpy/scipy 编译中 | 正常，耐心等；或再 pin 更低版本 |
| App 启动后立刻 `secrets.toml not found` | 代码直接 `st.secrets[xxx]` 导致异常 | 我们的 `_read_secret()` 已做 try/except，若仍报错请反馈 |
| 实时行情拉取失败 | Streamlit Cloud 出口 IP 被 Yahoo 限流 | 自动降级到 Part 1 fallback，不影响 demo |

## 关闭/重新部署

- **暂停应用**（不删）：Streamlit Cloud 控制台 → App → 汉堡菜单 → `Reboot` / `Delete`
- **更新代码**：本地 `git push` 即可，云端自动同步
- **换 LLM key**：控制台 → App → Settings → Secrets → 改完保存 → 自动 Reboot

---

有任何一步卡住，把报错截图/贴文本给我，我马上帮你定位。
