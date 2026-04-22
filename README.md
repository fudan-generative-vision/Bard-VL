# BARD-VL Project Page

这个目录现在用于 `BARD-VL` project page。本地页面素材已经放在 `assets/`，页面内容主要由 `site-config.js` 驱动，适合继续在当前目录里迭代文案、链接和媒体资源。

## 模板特点

- 纯静态：`index.html + styles.css + site-config.js + script.js`
- 不需要 React / Vue / 打包工具
- 适合部署到 GitHub Pages
- 支持三种媒体来源
  - 本地 `assets/`
  - 外链 CDN
  - YouTube iframe
- 自带几种常用版块
  - Hero 标题区
  - 作者与机构
  - 论文按钮
  - Abstract
  - Framework Figure
  - 视频画廊
  - Baseline vs Ours 对比
  - BibTeX

## 目录结构

```text
project-page-template/
├── .github/workflows/deploy.yml
├── assets/
├── index.html
├── README.md
├── script.js
├── site-config.js
└── styles.css
```

## 你主要改哪里

优先改这一个文件：

```text
site-config.js
```

你需要填的内容主要有：

- `site.title`
- `site.subtitle`
- `authors`
- `affiliations`
- `links`
- `hero.media`
- `sections`
- `citation.bibtex`

## 媒体怎么放

最简单的做法是把图片和视频直接放进：

```text
assets/
```

然后在 `site-config.js` 里引用：

```js
media: {
  type: "video",
  src: "assets/teaser.mp4",
  poster: "assets/teaser.jpg"
}
```

图片示例：

```js
media: {
  type: "image",
  src: "assets/framework.png",
  alt: "Method overview"
}
```

YouTube 示例：

```js
media: {
  type: "embed",
  src: "https://www.youtube.com/embed/YOUR_VIDEO_ID"
}
```

## 本地预览

在当前目录执行：

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```

## GitHub Pages 部署流程

### 方式一：推荐，自动部署

1. 新建一个 GitHub 仓库，例如 `my-project-page`
2. 把本目录所有文件推到仓库根目录
3. 默认分支使用 `main`
4. 推送后，GitHub Actions 会自动执行 `.github/workflows/deploy.yml`
5. 在 GitHub 仓库页面进入 `Settings > Pages`
6. `Source` 选择 `GitHub Actions`
7. 等待 workflow 成功
8. 页面地址通常是：

```text
https://<你的用户名>.github.io/<仓库名>/
```

### 方式二：最简单，直接从分支发布

如果你不想用 Actions，也可以：

1. 删除 `.github/workflows/deploy.yml`
2. 把静态文件保留在仓库根目录
3. 进入 `Settings > Pages`
4. `Source` 选择 `Deploy from a branch`
5. Branch 选 `main`，文件夹选 `/ (root)`

不过如果后面你想加构建步骤，还是推荐保留 Actions 方案。

## 自定义域名

如果你要绑定自己的域名，比如 `project.example.com`：

1. 在仓库根目录新建 `CNAME`
2. 文件里只写一行：

```text
project.example.com
```

3. 在域名服务商那里给这个域名配置到 GitHub Pages
4. GitHub `Settings > Pages` 中填写同样的自定义域名

## 针对学术项目页的建议

- 首屏只放最关键的信息：标题、作者、论文入口、一个强 teaser
- 视频不要太多，一屏展示 3 到 6 个最佳案例足够
- 每个 section 的标题要明确，不要只写 `Results`
- 对比区一定要有 `Baseline` 和 `Ours` 的显式标签
- 大视频建议放 CDN，不要把仓库塞满几百 MB
- 论文页不是论文正文，解释要短，证据要强

## 从 Hallo3 学什么

你给的页面本质上是下面这套模式：

- 顶部标题区
- 作者与机构
- Paper / Code / HuggingFace 按钮
- 多组视频结果画廊
- Abstract
- BibTeX

这套模式用静态模板完全可以实现，不一定要照搬 Vue 打包工程。
