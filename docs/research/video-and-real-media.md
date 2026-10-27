# 真实素材与低 token 视频方案核查

核查日期：2026-10-28。本轮研究没有付费模型调用，没有下载模型权重。测试素材、题目和参考答案在读取图片、PDF 页和视频帧后冻结；模型结果不能反向修改标签。

## 接口与费用边界

[DeepSeek 官方 Vision 文档](https://api-docs.deepseek.com/guides/vision/)提供 `image_url` 和多图输入；Responses 接口对应 `input_image`。截至核查时，这些接口文档未列出原生视频输入类型，因此项目采用本地解码、关键帧与时间戳后传多图的方式。不能将其他模型的 `type=video` 请求体直接用于 DeepSeek。

官方说明每张图独立计算视觉 token，上限为 1,024；`low` 先缩至 512×512，`high`/`original` 保持原图，`auto` 当前等同 `original`。多帧减少有直接成本意义；减少 JPEG 文件大小并不等价于同比减少视觉 token。表格小字可能需要保留分辨率，不能只优化缩图比例。

[官方价格页](https://api-docs.deepseek.com/quick_start/pricing/)在核查时将 `deepseek-flash` 映射至 `DeepSeek-V4.1-Flash-1017`；Pro 不支持视觉。Flash 高峰每百万输入 token 缓存未命中 0.30 美元、命中 0.006 美元，输出 1.20 美元；低谷为一半。离线预算用高峰未命中单价，实际报告依据服务端 usage；未知缓存命中或未返回 usage 的调用不应记为零成本。价格可能变化，运行前应重新核对。

## GitHub 参考与取舍

| 项目与固定提交 | 核查内容 | 本项目取舍 |
| --- | --- | --- |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/tree/36ca0229a898d6a25965f5f0b94c109c674833df) `36ca0229…`，BSD-3-Clause | [ContentDetector 源码](https://github.com/Breakthrough/PySceneDetect/blob/36ca0229a898d6a25965f5f0b94c109c674833df/scenedetect/detectors/content_detector.py)计算相邻帧 HSV 变化，含最短镜头约束；Python/OpenCV/NumPy，无模型权重要求。 | 借鉴本地场景变化评分和最小间隔，用 CPU 降低重复帧；不把单一差分峰当成完整事件理解。 |
| [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3/tree/05cee23deb854fd6ff82dcff1243a4f682e4d617) `05cee23d…`，仓库 Apache-2.0 | [模型代码](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/05cee23deb854fd6ff82dcff1243a4f682e4d617/videollama3/model/videollama3_arch.py)包含基于相邻像素差的 video token compressor；[README](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/05cee23deb854fd6ff82dcff1243a4f682e4d617/README.md)提供 `fps`、`max_frames`，示例依赖 PyTorch≥2.4、CUDA≥11.8、Transformers≥4.46.3，并使用 FlashAttention。 | 借鉴冗余压缩和硬帧数上限；本轮不部署 2B/7B 权重，不引入 CUDA/远端自定义模型代码。模型内部 token 压缩不能直接当成外部 API 已节省的 token。 |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da) `96588727…`，仓库 Apache-2.0 | 原 `Qwen2.5-VL` 仓库地址已重定向；这里明确引用当前 Qwen3-VL 提交。[vision_process.py](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)有帧数、FPS、总像素预算和视频元数据。推理使用 Transformers/PyTorch；视频后端支持 TorchVision、Decord、TorchCodec，TorchCodec 依赖 FFmpeg。 | 借鉴同时控制帧数和分辨率、保留时间信息。不复制其视觉 token 公式到 DeepSeek，也不引入大型 GPU 栈。 |

以上为设计参考，未复制第三方源代码，也未将项目代码许可视为其演示媒体或权重的自动授权。README 的 CUDA 示例不是对最低显存的证明；本轮没有测量这些模型的本地显存/速度。

## 冻结素材与许可

完整 URL、字节数、作者、许可、源版本、SHA-256、页码、题目和标签见 [real_media_cases.json](../../tests/fixtures/real_media_cases.json)。二进制仅缓存于 `.cache/media`，不提交仓库。

| 素材 | 许可依据 | 用例与局限 |
| --- | --- | --- |
| coffee.png，Rachel Michetti / Pikolo Espresso Bar | [scikit-image 对该照片的明确说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)：CC0。下载固定 scikit-image 提交 `6b22cec0d2fc154bc25c508675fe76f6f4e1402e`。 | 杯/勺数量、杯柄与匙碗方向、材质，多条件空间关系。属于小型真实照片细节诊断，不代表拥挤场景。 |
| rocket.jpg，SpaceX | [scikit-image 对该照片的明确说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.rocket)：public domain；同一固定提交。 | 边缘裁切的四座塔、左右计数、夜间照明、不能凭静帧断定起飞。 |
| USDA《Crop Production, October 2024》PDF，第 6、9 页 | [原报告](https://www.nass.usda.gov/Publications/Todays_Reports/reports/crop1024.pdf)标明由 USDA NASS Agricultural Statistics Board 发布；所选统计表页署 USDA NASS，无第三方版权标记。[USDA 版权政策](https://www.usda.gov/about-usda/policies-and-links)允许复制公共领域内容并要求保留署名，同时明确第三方材料例外。 | 第 6 页玉米跨年/跨月份多层表头、面积/单产/产量单位和 Other States 脚注；第 9 页水稻上下两表、磅与 cwt、品种总和和甜米脚注。不因为 `.gov` 域名就推定任何材料都可自由使用。署名：U.S. Department of Agriculture, National Agricultural Statistics Service。 |
| Big Buck Bunny trailer，W3C 镜像，约 32.44 秒 | [Blender 官方项目许可](https://peach.blender.org/about/)为 CC-BY-3.0，影片结尾也显示 Creative Commons Attribution 3.0。保留署名：`(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org`。 | 是动画，不是实拍。两个不重叠窗口分别检验重复切镜及动作/物体顺序；同一片源，不是两个独立视频样本。未使用其他项目中授权不清的演示片。 |

USDA 公共领域依据限定于上述政府署名统计表，不扩展到政府商标、第三方照片或整个网站，也不表示政府认可本项目。

## 可复现准备与离线约束

```powershell
# 首次明确允许下载；总源文件约 13 MB
python scripts/media_assets.py --download
# 缓存存在后不联网，逐项核验源 SHA 和派生页面 provenance
python scripts/media_assets.py
# 普通单元测试全部使用 MockTransport，不联网、不解码视频、不调用模型
python -m pytest tests/test_media_assets.py -q --basetemp .tmp/pytest-media-assets
```

`media_assets.py` 只接受固定白名单主机的 HTTPS，拒绝重定向、URL 凭据、非标准端口和路径穿越；下载前检查声明大小、流式检查实际大小，完整 SHA 匹配才原子写入。缺失缓存默认失败，已有源文件校验失败也不会自动覆盖。单个源最多 32 MiB，manifest 对各素材给出更紧上限。

PDF 使用本地 Poppler `pdftoppm`，固定 130 dpi 和原始页码，无裁剪。参考环境为 Poppler 26.07.0，参考 PNG hash 已记录。不同 Poppler/字体版本可能生成不同 PNG 字节，因此严格校验源 PDF，派生 PNG 的实际 hash、源 hash、页码、dpi、渲染器版本写入 `.provenance.json` 并在复用时校验。跨机器报告应记录实际 PNG hash，不假装字节级一致。准备脚本不下载 Poppler；缺失时明确报错。

视频不制作派生片段，素材准备阶段两题都返回同一完整 MP4；[素材清单](../validation/real-media-assets-2026-10-28.json)记录这一点。实际分析可使用问题中明确写出的时间窗作为请求参数，源文件相同不代表处理范围相同。动作标签来自预先查看的 0.5 秒间隔帧，而非模型描述。临时接触表只辅助人工核验，不作为模型输入。

## 评分与可得结论

1. 照片同时检查数量、空间关系和无法确认项；表格逐个检查冻结的 numeric_cells、单位与脚注。只命中几个关键词不能代表整题通过。
2. 视频预先固定 `[13,17]`、`[19,23]` 秒题目范围。`start_seconds/end_seconds` 来自问题文本，可作为请求参数；更细的 `evidence_intervals` 和 `expected_sequence` 仅用于评分，不能给抽帧器或模型。本轮比较 baseline 的全片均匀 12 帧/original 与 optimized 的题目时间窗自适应最多 6 帧/low，是范围、采样、帧数和分辨率共同变化的组合方案。不能把全部差异单独归因于 adaptive；若分析算法差异，应另做相同时间窗与帧数的离线消融。报告需保留两侧实际分析范围、选中时间戳、事件覆盖率及遗漏区间。
3. 时间顺序需检查实际回答，关键词组仅提供覆盖辅助。“不能确认已经离地”等否定句不能被 forbidden_claims 的原始子串误判为幻觉。
4. baseline/optimized 应使用同一文件、问题、输出上限和模型，分别记录实际 usage、费用、完整性。字节压缩率、字符估计、服务端 token 是不同指标。
5. 六题是功能诊断，不足以得出通用图像/视频识别质量结论。动画镜头切换较快；少帧策略可能漏掉约半秒的事件，明确承认未知比编造完整顺序更可取。尚不覆盖实拍动作、长视频、多语言字幕或音频理解。

测试执行结果写入本轮验证报告；本研究文档不将方案建议当作已测得收益。
