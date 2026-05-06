# Third-party code (not redistributed in full in some course releases)

This directory is reserved for **local clones** of upstream projects used by baseline inference:

| Directory | Upstream | Purpose |
|-----------|----------|---------|
| `fc-clip/` | [bytedance/fc-clip](https://github.com/bytedance/fc-clip) | FC-CLIP Detectron2 evaluation |
| `SED/` | [xb534/SED](https://github.com/xb534/SED) | SED Detectron2 evaluation |

**Obtain code:** `git clone` the official repositories into these folder names, then follow each project’s `INSTALL.md` and compile CUDA extensions (FC-CLIP) as required.

**Weights:** download from the links in each paper’s / project’s README (e.g. Google Drive). This course release may omit large `*.pth` files from the zip.

---

本目录用于放置 **官方仓库的本地 clone**，供 baseline 推理使用。请自行 `git clone` 并安装依赖；权重请从原作者提供的链接下载，本作业压缩包可能 **不包含** 大文件权重。
