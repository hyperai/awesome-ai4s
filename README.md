# Awesome AI for Science

 [**前言**](#前言)
- [**AI+   生物医药：AI+Biopharmaceutical**](#ai---生物医药aibiopharmaceutical)
  - [**1. AdaDR 在药物重定位方面的性能优于多个基准方法**](#1-adadr-在药物重定位方面的性能优于多个基准方法)
  - [**2. IMN4NPD 加快分子网络中广泛集群的去复制，对自循环与成对节点提供标注**](#2-imn4npd-加快分子网络中广泛集群的去复制对自循环与成对节点提供标注)
  - [**3. 深度生成模型 MIDAS 用于单细胞多组学数据马赛克整合**](#3-深度生成模型-midas-用于单细胞多组学数据马赛克整合)
  - [**4. 基于蛋白质口袋的 3D 分子生成模型——ResGen**](#4-基于蛋白质口袋的-3d-分子生成模型resgen)
  - [**5. 大模型 + 机器学习高精度预测酶动力学参数**](#5-大模型--机器学习高精度预测酶动力学参数)
  - [**6. MIT 利用深度学习发现新型抗生素**](#6-mit-利用深度学习发现新型抗生素)
  - [**7. 神经网络解密 GPCR-G 蛋白偶联选择性**](#7-神经网络解密-gpcr-g-蛋白偶联选择性)
  - [**8. Macformer 将无环药物菲卓替尼大环化**](#8-macformer-将无环药物菲卓替尼大环化)
  - [**9. 快速自动扫描套件 FAST 高效获取样本信息**](#9-快速自动扫描套件-fast-高效获取样本信息)
  - [**10. 回归网络 + CGMD，预测百亿种多肽的自组装特性**](#10-回归网络--cgmd预测百亿种多肽的自组装特性)
  - [**11. 无监督学习预测 7100 万种基因突变**](#11-无监督学习预测-7100-万种基因突变)
  - [**12. 基于图神经网络 (GNN) 开发气味分析 AI**](#12-基于图神经网络-gnn-开发气味分析-ai)
  - [**13. 图神经网络筛选安全高效的抗衰老成分**](#13-图神经网络筛选安全高效的抗衰老成分)
  - [**14. 机器学习量化分析多巴胺的释放量和释放位置**](#14-机器学习量化分析多巴胺的释放量和释放位置)
  - [**15. 机器学习发现三种抗衰老药物**](#15-机器学习发现三种抗衰老药物)
  - [**16. 深度学习筛选抑制鲍曼不动杆菌的新型抗生素**](#16-深度学习筛选抑制鲍曼不动杆菌的新型抗生素)
  - [**17. 机器学习模型应用于预测生物墨水可打印性**](#17-机器学习模型应用于预测生物墨水可打印性)
  - [**18. 机器学习分化多能干细胞**](#18-机器学习分化多能干细胞)
  - [**19. 机器学习模型预测长效注射剂药物释放速率**](#19-机器学习模型预测长效注射剂药物释放速率)
  - [**20. 机器学习算法有效预测植物抗疟性**](#20-机器学习算法有效预测植物抗疟性)
  - [**21. 机器学习集成方法预测病毒蛋白片段免疫原性**](#21-机器学习集成方法预测病毒蛋白片段免疫原性)
  - [**22. 用生成式 AI 开发新型抗生素**](#22-用生成式-ai-开发新型抗生素)
  - [**23. 基于深度学习研发一种自动化、高速、多维的单粒子追踪系统**](#23-基于深度学习研发一种自动化高速多维的单粒子追踪系统)
  - [**24. ProEnsemble 机器学习框架：优化进化通路启动子组合**](#24-proensemble-机器学习框架优化进化通路启动子组合)
  - [**25. 微环境感知图神经网络 ProtLGN 指导蛋白质定向进化**](#25-微环境感知图神经网络-protlgn-指导蛋白质定向进化)
  - [**26. 深度学习模型 AlphaPPIMd：用于蛋白质-蛋白质复合物构象集合探索**](#26-深度学习模型-alphappimd用于蛋白质-蛋白质复合物构象集合探索)
  - [**27. 新型肿瘤抑制蛋白降解剂 dp53m 可抑制癌细胞增殖**](#27-新型肿瘤抑制蛋白降解剂-dp53m-可抑制癌细胞增殖)
  - [**28. CVPR 最佳学生论文！多模态模型 BioCLIP 实现零样本学习**](#28-cvpr-最佳学生论文多模态模型-bioclip-实现零样本学习)
  - [**29. 1 亿参数！细胞大模型 scFoundation 可对 2 万基因同时建模**](#29-1-亿参数细胞大模型-scfoundation-可对-2-万基因同时建模)
  - [**30. 入选顶会 ICML，蛋白质语言模型 ESM-AA 超越传统 SOTA**](#30-入选顶会-icml蛋白质语言模型-esm-aa-超越传统-sota)
  - [**31. SPACE 算法登 Cell 子刊！组织模块发现能力领先同类工具**](#31-space-算法登-cell-子刊组织模块发现能力领先同类工具)
  - [**32. 基于 AlphaFold 实现新突破，揭示蛋白质动态多样性**](#32-基于-alphafold-实现新突破揭示蛋白质动态多样性)
  - [**33. 基于扩散模型开发 P450 酶从头设计方法 P450Diffusion**](#33-基于扩散模型开发-p450-酶从头设计方法-p450diffusion)
  - [**34. 将等变图神经网络用于靶蛋白结合位点预测，性能提升 20%**](#34-将等变图神经网络用于靶蛋白结合位点预测性能提升-20)
  - [**35. 20 个实验数据创造 AI 蛋白质里程碑！FSFP 有效优化蛋白质预训练模型**](#35-20-个实验数据创造-ai-蛋白质里程碑fsfp-有效优化蛋白质预训练模型)
  - [**36. 可迁移深度学习模型鉴定多类型 RNA 修饰、显著减少计算成本**](#36-可迁移深度学习模型鉴定多类型-rna-修饰显著减少计算成本)
  - [**37. InstructProtein：利用知识指令对齐蛋白质语言与人类语言**](#37-instructprotein利用知识指令对齐蛋白质语言与人类语言)
  - [**38. 蛋白质-文本生成框架 ProtT3 实现蛋白质数据与文本信息跨模态解读**](#38-蛋白质-文本生成框架-prott3-实现蛋白质数据与文本信息跨模态解读)
  - [**39. CPDiffusion 模型，超低成本、全自动设计功能型蛋白质**](#39-cpdiffusion-模型超低成本全自动设计功能型蛋白质)
  - [**40. 基于蛋白质语言模型和密集检索技术，一种全新的蛋白质同源物检测方法**](#40-基于蛋白质语言模型和密集检索技术一种全新的蛋白质同源物检测方法)
  - [**41. AlphaProteo 可高效设计靶蛋白结合物，亲和力提高 300 倍**](#41-alphaproteo-可高效设计靶蛋白结合物亲和力提高-300-倍)
  - [**42. 全新去噪蛋白质语言模型 DePLM，突变效应预测优于 SOTA 模型**](#42-全新去噪蛋白质语言模型-deplm突变效应预测优于-sota-模型)
  - [**43. 几何深度生成模型 DynamicBind，实现蛋白质动态对接预测**](#43-几何深度生成模型-dynamicbind实现蛋白质动态对接预测)
- [**AI+   医疗健康：AI+Healthcare**](#ai---医疗健康aihealthcare)
  - [**1. 深度学习系统 DeepDR Plus 用眼底图像预测糖尿病视网膜病变**](#1-深度学习系统-deepdr-plus-用眼底图像预测糖尿病视网膜病变)
  - [**2. 逻辑回归模型分析高绿色景观指数可降低 MetS 风险**](#2-逻辑回归模型分析高绿色景观指数可降低-mets-风险)
  - [**3. 深度学习系统助力初级眼科医生的诊断一致性提高 12%**](#3-深度学习系统助力初级眼科医生的诊断一致性提高-12)
  - [**4. GSP-GCNs 实现帕金森病诊断准确率高达 90.2%**](#4-gsp-gcns-实现帕金森病诊断准确率高达-902)
  - [**5. 乳腺癌预后评分系统 MIRS**](#5-乳腺癌预后评分系统-mirs)
  - [**6. 视网膜图像基础模型 RETFound，预测多种系统性疾病**](#6-视网膜图像基础模型-retfound预测多种系统性疾病)
  - [**7. SVM 优化触觉传感器，盲文识别率达 96.12%**](#7-svm-优化触觉传感器盲文识别率达-9612)
  - [**8. 中科院基因组所建立开放生物医学成像档案**](#8-中科院基因组所建立开放生物医学成像档案)
  - [**9. AI Lunit 阅读乳腺 X 光片的准确率与医生相当**](#9-ai-lunit-阅读乳腺-x-光片的准确率与医生相当)
  - [**10. 特征选择策略检测乳腺癌生物标志物**](#10-特征选择策略检测乳腺癌生物标志物)
  - [**11. 梯度提升机模型准确预测 BPSD 亚综合征**](#11-梯度提升机模型准确预测-bpsd-亚综合征)
  - [**12. 机器学习模型预测患者一年内死亡率**](#12-机器学习模型预测患者一年内死亡率)
  - [**13. AI 新脑机技术让失语患者「开口说话」**](#13-ai-新脑机技术让失语患者开口说话)
  - [**14. 基于深度学习的胰腺癌人工智能检测**](#14-基于深度学习的胰腺癌人工智能检测)
  - [**15. 机器学习辅助肺癌筛查的群体有效性**](#15-机器学习辅助肺癌筛查的群体有效性)
  - [**16. 卵巢癌诊断人工智能融合模型 MCF，输入常规实验室检验数据和年龄即可计算卵巢癌的患病风险**](#16-卵巢癌诊断人工智能融合模型-mcf输入常规实验室检验数据和年龄即可计算卵巢癌的患病风险)
  - [**17. 谷歌发布 HEAL 架构，4 步评估医学 AI 工具是否公平**](#17-谷歌发布-heal-架构4-步评估医学-ai-工具是否公平)
  - [**18. 借鉴语义分割，开发空间转录组语义注释工具 Pianno**](#18-借鉴语义分割开发空间转录组语义注释工具-pianno)
  - [**19. AI 模型 UniFMIR，突破现有荧光显微成像极限**](#19-ai-模型-unifmir突破现有荧光显微成像极限)
  - [**20. 深度学习系统，提高癌症生存预测准确性**](#20-深度学习系统提高癌症生存预测准确性)
  - [**21. MemSAM 将「分割一切」模型用于医学视频分割**](#21-memsam-将分割一切模型用于医学视频分割)
  - [**22. 医学图像分割模型 Medical SAM 2 刷新医学图像分割 SOTA 榜**](#22-医学图像分割模型-medical-sam-2-刷新医学图像分割-sota-榜)
  - [**23. 机器学习抗击化疗耐药性与肿瘤复发，构筑乳腺癌干细胞的有力防线**](#23-机器学习抗击化疗耐药性与肿瘤复发构筑乳腺癌干细胞的有力防线)
  - [**24. 糖尿病诊疗的视觉-大语言模型 DeepDR-LLM 登 Nature 子刊**](#24-糖尿病诊疗的视觉-大语言模型-deepdr-llm-登-nature-子刊)
  - [**25. 水平直逼高级病理学家！清华团队提出 AI 基础模型 ROAM，实现胶质瘤精准诊断**](#25-水平直逼高级病理学家清华团队提出-ai-基础模型-roam实现胶质瘤精准诊断)
  - [**26. 医学图像分割通用模型 ScribblePrompt，性能优于 SAM**](#26-医学图像分割通用模型-scribbleprompt性能优于-sam)
  - [**27. 数字孪生脑平台，展现出类似人脑中观测的临界现象与相似认知功能**](#27-数字孪生脑平台展现出类似人脑中观测的临界现象与相似认知功能)
  - [**28. 自动化大模型对话 Agent 模拟系统，可初诊抑郁症**](#28-自动化大模型对话-agent-模拟系统可初诊抑郁症)
  - [**29. 深度学习模型 LucaProt，助力 RNA 病毒识别**](#29-深度学习模型-lucaprot助力-rna-病毒识别)
  - [**30. 医学图像预训练框架 UniMedI，打破医学数据异构化藩篱**](#30-医学图像预训练框架-unimedi打破医学数据异构化藩篱)
  - [**31. 多语言医学大模型 MMed-Llama 3，更加适配医疗应用场景**](#31-多语言医学大模型-mmed-llama-3更加适配医疗应用场景)
  - [**32. 胶囊内窥镜图像拼接方法 S2P-Matching，助力胶囊内窥镜图像拼接**](#32-胶囊内窥镜图像拼接方法-s2p-matching助力胶囊内窥镜图像拼接)
- [**AI+ 材料化学：AI+Materials Chemistry**](#ai-材料化学aimaterials-chemistry)
  - [**1. 高通量计算框架 33 分钟生成 12 万种新型 MOFs 候选材料**](#1-高通量计算框架-33-分钟生成-12-万种新型-mofs-候选材料)
  - [**2. 机器学习算法模型筛选 P-SOC 电极材料**](#2-机器学习算法模型筛选-p-soc-电极材料)
  - [**3. SEN 机器学习模型，实现高精度的材料性能预测**](#3-sen-机器学习模型实现高精度的材料性能预测)
  - [**4. 深度学习工具 GNoME 发现 220 万种新晶体**](#4-深度学习工具-gnome-发现-220-万种新晶体)
  - [**5. 场诱导递归嵌入原子神经网络可准确描述外场强度、方向变化**](#5-场诱导递归嵌入原子神经网络可准确描述外场强度方向变化)
  - [**6. 机器学习预测多孔材料水吸附等温线**](#6-机器学习预测多孔材料水吸附等温线)
  - [**7. 利用机器学习优化 BiVO(4) 光阳极的助催化剂**](#7-利用机器学习优化-bivo4-光阳极的助催化剂)
  - [**8. RetroExplainer 算法基于深度学习进行逆合成预测**](#8-retroexplainer-算法基于深度学习进行逆合成预测)
  - [**9. 深度神经网络+自然语言处理，开发抗蚀合金**](#9-深度神经网络自然语言处理开发抗蚀合金)
  - [**10. 深度学习通过表面观察确定材料的内部结构**](#10-深度学习通过表面观察确定材料的内部结构)
  - [**11. 利用创新 X 射线闪烁体开发 3 种新材料**](#11-利用创新-x-射线闪烁体开发-3-种新材料)
  - [**12. 半监督学习提取无标签数据中的隐藏信息**](#12-半监督学习提取无标签数据中的隐藏信息)
  - [**13. 基于自动机器学习进行知识自动提取**](#13-基于自动机器学习进行知识自动提取)
  - [**14. 一种三维 MOF 材料吸附行为预测的机器学习模型 Uni-MOF**](#14-一种三维-mof-材料吸附行为预测的机器学习模型-uni-mof)
  - [**15. 微电子加速迈向后摩尔时代！集成 DNN 与纳米薄膜技术，精准分析入射光角度**](#15-微电子加速迈向后摩尔时代集成-dnn-与纳米薄膜技术精准分析入射光角度)
  - [**16. 重塑锂电池性能边界，基于集成学习提出简化电化学模型**](#16-重塑锂电池性能边界基于集成学习提出简化电化学模型)
  - [**17. 最强铁基超导磁体诞生！基于机器学习，磁场强度超过先前记录 2.7 倍**](#17-最强铁基超导磁体诞生基于机器学习磁场强度超过先前记录-27-倍)
  - [**18. 神经网络替代密度泛函理论！通用材料模型实现超精准预测**](#18-神经网络替代密度泛函理论通用材料模型实现超精准预测)
  - [**19. 神经网络密度泛函框架打开物质电子结构预测的黑箱**](#19-神经网络密度泛函框架打开物质电子结构预测的黑箱)
  - [**20. 用神经网络首创全前向智能光计算训练架构，国产光芯片实现重大突破**](#20-用神经网络首创全前向智能光计算训练架构国产光芯片实现重大突破)
  - [**21. 化学大语言模型 ChemLLM 覆盖 7 百万问答数据，专业能力比肩 GPT-4**](#21-化学大语言模型-chemllm-覆盖-7-百万问答数据专业能力比肩-gpt-4)
  - [**22. 可晶圆级生产的人工智能自适应微型光谱仪**](#22-可晶圆级生产的人工智能自适应微型光谱仪)
  - [**23. GNNOpt 模型，识别数百种太阳能电池和量子候选材料**](#23-gnnopt-模型识别数百种太阳能电池和量子候选材料)
- [**AI+动植物科学：AI+Zoology-Botany**](#ai动植物科学aizoology-botany)
  - [**1. SBeA 基于少样本学习框架进行动物社会行为分析**](#1-sbea-基于少样本学习框架进行动物社会行为分析)
  - [**2. 基于孪生网络的深度学习方法，自动捕捉胚胎发育过程**](#2-基于孪生网络的深度学习方法自动捕捉胚胎发育过程)
  - [**3. 利用无人机采集植物表型数据的系统化流程，预测最佳采收日期**](#3-利用无人机采集植物表型数据的系统化流程预测最佳采收日期)
  - [**4. AI 相机警报系统准确区分老虎和其他物种**](#4-ai-相机警报系统准确区分老虎和其他物种)
  - [**5. 利用拉布拉多猎犬数据，对比 3 种模型，发现了影响嗅觉检测犬表现的行为特性**](#5-利用拉布拉多猎犬数据对比-3-种模型发现了影响嗅觉检测犬表现的行为特性)
  - [**6. 基于人脸识别 ArcFace Classification Head 的多物种图像识别模型**](#6-基于人脸识别-arcface-classification-head-的多物种图像识别模型)
  - [**7. 利用 Python API 与计算机视觉 API，监测日本的樱花开放情况**](#7-利用-python-api-与计算机视觉-api监测日本的樱花开放情况)
  - [**8. 基于机器学习的群体遗传方法，揭示葡萄风味的形成机制**](#8-基于机器学习的群体遗传方法揭示葡萄风味的形成机制)
  - [**9. 综述：借助 AI 更高效地开启生物信息学研究**](#9-综述借助-ai-更高效地开启生物信息学研究)
  - [**10. BirdFlow 模型准确预测候鸟的飞行路径**](#10-birdflow-模型准确预测候鸟的飞行路径)
  - [**11. 新的鲸鱼生物声学模型，可识别 8 种鲸类**](#11-新的鲸鱼生物声学模型可识别-8-种鲸类)
- [**AI+农林畜牧业：AI+Agriculture-Forestry-Animal husbandry**](#ai农林畜牧业aiagriculture-forestry-animal-husbandry)
  - [**1. 利用卷积神经网络，对水稻产量进行迅速、准确的统计**](#1-利用卷积神经网络对水稻产量进行迅速准确的统计)
  - [**2. 通过 YOLOv5 算法，设计监测母猪姿势与猪仔出生的模型**](#2-通过-yolov5-算法设计监测母猪姿势与猪仔出生的模型)
  - [**3. 结合实验室观测与机器学习，证明番茄与烟草植物在胁迫环境下发出的超声波能在空气中传播**](#3-结合实验室观测与机器学习证明番茄与烟草植物在胁迫环境下发出的超声波能在空气中传播)
  - [**4. 无人机+ AI 图像分析，检测林业害虫**](#4-无人机-ai-图像分析检测林业害虫)
  - [**5. 计算机视觉+深度学习开发奶牛跛行检测系统**](#5-计算机视觉深度学习开发奶牛跛行检测系统)
- [**AI+ 气象学：AI+Meteorology**](#ai-气象学aimeteorology)
  - [**1. 综述：数据驱动的机器学习天气预报模型**](#1-综述数据驱动的机器学习天气预报模型)
  - [**2. 综述：从雹暴中心收集数据，利用大模型预测极端天气**](#2-综述从雹暴中心收集数据利用大模型预测极端天气)
  - [**3. 利用全球风暴解析模拟与机器学习，创建新算法，准确预测极端降水**](#3-利用全球风暴解析模拟与机器学习创建新算法准确预测极端降水)
  - [**4. 基于随机森林的机器学习模型 CSU-MLP，预测中期恶劣天气**](#4-基于随机森林的机器学习模型-csu-mlp预测中期恶劣天气)
- [**AI+ 天文学：AI+Astronomy**](#ai-天文学aiastronomy)
  - [**1. PRIMO 算法学习黑洞周围的光线传播规律，重建出更清晰的黑洞图像**](#1-primo-算法学习黑洞周围的光线传播规律重建出更清晰的黑洞图像)
  - [**2. 利用模拟数据训练计算机视觉算法，对天文图像进行锐化「还原」**](#2-利用模拟数据训练计算机视觉算法对天文图像进行锐化还原)
  - [**3. 利用无监督机器学习算法 Astronomaly ，找到了之前为人忽视的异常现象**](#3-利用无监督机器学习算法-astronomaly-找到了之前为人忽视的异常现象)
  - [**4. 基于机器学习的 CME 识别与参数获取方法**](#4-基于机器学习的-cme-识别与参数获取方法)
  - [**5. 深度学习发现 107 例中性碳吸收线**](#5-深度学习发现-107-例中性碳吸收线)
  - [**6. StarFusion 模型实现高空间分辨率图像的预测**](#6-starfusion-模型实现高空间分辨率图像的预测)
- [**AI+ 自然灾害：AI+Natural Disaster**](#ai-自然灾害ainatural-disaster)
  - [**1. 机器学习预测未来 40 年的地面沉降风险**](#1-机器学习预测未来-40-年的地面沉降风险)
  - [**2. 语义分割模型 SCDUNet++ 用于滑坡测绘**](#2-语义分割模型-scdunet-用于滑坡测绘)
  - [**3. 神经网络将太阳二维图像转为三维重建图像**](#3-神经网络将太阳二维图像转为三维重建图像)
  - [**4. 可叠加神经网络分析自然灾害中的影响因素**](#4-可叠加神经网络分析自然灾害中的影响因素)
  - [**5. 利用可解释性 AI ，分析澳大利亚吉普斯兰市的不同地理因素**](#5-利用可解释性-ai-分析澳大利亚吉普斯兰市的不同地理因素)
  - [**6. 基于机器学习的洪水预报模型**](#6-基于机器学习的洪水预报模型)
  - [**7. ED-DLSTM实现无监测数据地区洪水预测**](#7-ed-dlstm实现无监测数据地区洪水预测)
  - [**8. ChloroFormer 模型提前预警海洋藻类爆发**](#8-chloroformer-模型提前预警海洋藻类爆发)
- [**AI4S 政策解读：AI4S Policy**](#ai4s-政策解读ai4s-policy)
  - [**1. 科技部出台政策防范学术界 AI 枪手**](#1-科技部出台政策防范学术界-ai-枪手)
  - [**2. 政策：科技部会同自然科学基金委启动「人工智能驱动的科学研究」( AI for Science ) 专项部署工作**](#2-政策科技部会同自然科学基金委启动人工智能驱动的科学研究-ai-for-science--专项部署工作)
- [**其他：Others**](#其他others)
  - [**1. TacticAI 足球助手战术布局实用性高达 90%**](#1-tacticai-足球助手战术布局实用性高达-90)
  - [**2. 去噪扩散模型 SPDiff 实现长程人流移动模拟**](#2-去噪扩散模型-spdiff-实现长程人流移动模拟)
  - [**3. 智能化科学设施推进科研范式变革**](#3-智能化科学设施推进科研范式变革)
  - [**4. DeepSymNet 基于监督学习来表示符号表达式**](#4-deepsymnet-基于监督学习来表示符号表达式)
  - [**5. 大语言模型 ChipNeMo 辅助工程师完成芯片设计**](#5-大语言模型-chipnemo-辅助工程师完成芯片设计)
  - [**6. AlphaGeometry 可解决几何学问题**](#6-alphageometry-可解决几何学问题)
  - [**7. 强化学习用于城市空间规划**](#7-强化学习用于城市空间规划)
  - [**8. ChatArena 框架，与大语言模型一起玩狼人杀**](#8-chatarena-框架与大语言模型一起玩狼人杀)
  - [**9. 综述：30 位学者合力发表 Nature，10 年回顾解构 AI 如何重塑科研范式**](#9-综述30-位学者合力发表-nature10-年回顾解构-ai-如何重塑科研范式)
  - [**10. Ithaca 协助金石学家进行文本修复、时间归因和地域归因的工作**](#10-ithaca-协助金石学家进行文本修复时间归因和地域归因的工作)
  - [**11. AI 在超光学中的正问题及逆问题、基于超表面系统的数据分析**](#11-ai-在超光学中的正问题及逆问题基于超表面系统的数据分析)
  - [**12. 一种新的地理空间人工智能方法：地理神经网络加权逻辑回归**](#12-一种新的地理空间人工智能方法地理神经网络加权逻辑回归)
  - [**13. 利用扩散模型生成神经网络参数，将时空少样本学习转变为扩散模型的预训练问题**](#13-利用扩散模型生成神经网络参数将时空少样本学习转变为扩散模型的预训练问题)
  - [**14. 李飞飞团队 AI4S 最新洞察：16 项创新技术汇总，覆盖生物/材料/医疗/问诊**](#14-李飞飞团队-ai4s-最新洞察16-项创新技术汇总覆盖生物材料医疗问诊)
  - [**15. 精准预测武汉房价！osp-GNNWR 模型准确描述复杂空间过程和地理现象**](#15-精准预测武汉房价osp-gnnwr-模型准确描述复杂空间过程和地理现象)
  - [**16. 首个海洋大语言模型 OceanGPT 入选 ACL 2024！水下具身智能成现实**](#16-首个海洋大语言模型-oceangpt-入选-acl-2024水下具身智能成现实)
  - [**17. 引入零样本学习，发布针对甲骨文破译优化的条件扩散模型**](#17-引入零样本学习发布针对甲骨文破译优化的条件扩散模型)
  - [**18. 斯坦福/苹果等 23 所机构发布 DCLM 基准测试，基础模型与 Llama3 8B 表现相当**](#18-斯坦福苹果等-23-所机构发布-dclm-基准测试基础模型与-llama3-8b-表现相当)
  - [**19. PoCo 解决数据源异构难题，实现机器人多任务灵活执行**](#19-poco-解决数据源异构难题实现机器人多任务灵活执行)
  - [**20. 含 14 万张图像！甲骨文数据集助力团队摘冠 ACL 最佳论文**](#20-含-14-万张图像甲骨文数据集助力团队摘冠-acl-最佳论文)
  - [**21. 用机器学习分离抹香鲸发音字母表，高度类似人类语言，信息承载能力更强**](#21-用机器学习分离抹香鲸发音字母表高度类似人类语言信息承载能力更强)
  - [**22. 基于预训练 LLM 提出信道预测方案，GPT-2 赋能无线通信物理层**](#22-基于预训练-llm-提出信道预测方案gpt-2-赋能无线通信物理层)
  - [**23. 首个多缝线刺绣生成对抗网络模型**](#23-首个多缝线刺绣生成对抗网络模型)
## **前言**

从 2020 年开始，以 AlphaFold 为代表的科研项目将 AI for Science (AI4S) 推向了 AI 应用的主舞台。近年来，从生物医药到天文气象、再到材料化学等基础学科，都成为了 AI 的新战场。

随着越来越多的交叉学科人才开始在其研究领域应用机器学习、深度学习等技术进行数据处理、构建模型，加之跨学科研究团队的合作日益加强，AI4S 的能力被更多科研人员所关注到，但却未达到规模化应用的目标。提高相关研究的可复用性、降低技术门槛、提高数据质量等诸多问题亟待解决。

目前，除了高校、科研机构在积极探索 AI4S 外，多国政府及头部科技企业也都关注到了 AI 革新科研的潜力，并进行了相关的政策疏导与布局，可以说 AI4S 已经是大势所趋。

作为最早一批关注到 AI for Science 的社区，「HyperAI超神经」在陪伴行业成长的同时，也乐于将最新的研究进展与成果进行普适化分享，我们希望通过解读前沿论文与政策的方式，令更多团队看到 AI 对于科研的帮助，为 AI for Science 的发展贡献力量。

目前，HyperAI超神经已经解读分享了近百篇论文，为了便于大家检索，我们将文章根据学科进行分类，并展示了发表期刊及时间，提取了关键词（研究团队、相关研究、数据集等），大家可以点击题目获取解读文章（内含完整论文下载链接），或者直接点击论文标题查看原文。

本文档将以开源项目的形式呈现，我们将持续更新解读文章，同时也欢迎大家投稿优秀研究成果，如果您所在的团队/课题组有报道需求，可添加微信：神经星星（微信号：Hyperai01）。

## **AI+   生物医药：AI+Biopharmaceutical**

### **1. [AdaDR 在药物重定位方面的性能优于多个基准方法](https://hyper.ai/news/30434)**

* **中文解读：** [https://hyper.ai/news/30434](https://hyper.ai/news/30434)

* **科研团队：** 中南大学李敏研究团队

* **相关研究：** Gdataset 数据集、Cdataset 数据集、Ldataset 数据集、LRSSL 数据集、GCNs 框架、AdaDR

* **发布期刊：** Bioinformatics, 2024.01

* **论文链接：** [Drug repositioning with adaptive graph convolutional networks](https://academic.oup.com/bioinformatics/article/40/1/btad748/7467059 
)

### **2. [IMN4NPD 加快分子网络中广泛集群的去复制，对自循环与成对节点提供标注](https://hyper.ai/news/30363)**

* **中文解读：** [https://hyper.ai/news/30363](https://hyper.ai/news/30363)

* **科研团队：** 中南大学刘韶研究团队

*  **相关研究：** MS/MS 光谱数据库、Structure 数据库、molDiscovery、NPClassifier、molDiscovery、t-SNE

* **发布期刊：** Analytical Chemistry, 2024.02

* **论文链接：** [IMN4NPD: An Integrated Molecular Networking Workflow for Natural Product Dereplication](https://doi.org/10.1021/acs.analchem.3c04746)

### **3. [深度生成模型 MIDAS 用于单细胞多组学数据马赛克整合](https://hyper.ai/news/29785)**

* **中文解读：** [https://hyper.ai/news/29785](https://hyper.ai/news/29785)

* **科研团队：** 军事医学研究院应晓敏研究团队

* **相关研究：** IPBMC  数据集、dogma-full 数据集、teadog-full 数据集、MMIDAS、self-supervised learning、information-theoretic approaches、深度神经网络、SGVB、单细胞多组学马赛克数据

* **发布期刊：** Nature Biotechnology, 2024.01

* **论文链接：** [Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS](https://www.nature.com/articles/s41587-023-02040-y)

### **4. [基于蛋白质口袋的 3D 分子生成模型——ResGen](https://hyper.ai/news/29026)**

* **中文解读：** [https://hyper.ai/news/29026](https://hyper.ai/news/29026)

* **科研团队：** 浙大侯廷军研究团队

* **相关研究：** CrossDock2020 数据集、全局自回归、原子自回归、并行多尺度建模、SBMG。比最优技术快 8 倍

* **发布期刊：** Nature Machine Intelligence, 2023.09

* **论文链接：** [ResGen is a pocket-aware 3D molecular generation model based on parallel multiscale modelling](https://www.nature.com/articles/s42256-023-00712-7)

### **5. [大模型 + 机器学习高精度预测酶动力学参数](https://hyper.ai/news/29000)**

* **中文解读：** [https://hyper.ai/news/29000](https://hyper.ai/news/29000)

* **科研团队：** 中科院罗小舟研究团队

* **相关研究：** kcat/Km  数据集、米氏常数数据集、pH 和温度数据集、DLKcat 数据集、UniKP 框架、ProtT5-XL-UniRef50、SMILES Transformer model、集成性模型、随机森林、极端随机树、线性回归模型

* **发布期刊：** Nature Communications, 2023.12

* **论文链接：** [UniKP: a unified framework for the prediction of enzyme kinetic parameters](https://www.nature.com/articles/s41467-023-44113-1)

### **6. [MIT 利用深度学习发现新型抗生素](https://hyper.ai/news/28886)**

* **中文解读：** [https://hyper.ai/news/28886](https://hyper.ai/news/28886)

* **科研团队：** MIT 研究团队

* **相关研究：** Mcule 数据库、Broad Institute 数据库、图神经网络 Chemprop、深度学习。筛选出 3,646 种抗生素化合物

* **发布期刊：** Nature, 2023.12

* **论文链接：** [Discovery of a structural class of antibiotics with explainable deep learning](https://www.nature.com/articles/s41586-023-06887-8)

### **7. [神经网络解密 GPCR-G 蛋白偶联选择性](https://hyper.ai/news/28361)**

* **中文解读：** [https://hyper.ai/news/28361](https://hyper.ai/news/28361)

* **科研团队：** 佛罗里达大学的研究团队

* **相关研究：** 二元分类神经网络、机器学习、无监督深度学习模型。建立了包括不同哺乳动物的 124 种 GPCRs 的粗粒度模型
* **发布期刊：** Cell Reports, 2023.09

* **论文链接：** [Rules and mechanisms governing G protein coupling selectivity of GPCRs](https://doi.org/10.1016/j.celrep.2023.113173)

### **8. [Macformer 将无环药物菲卓替尼大环化](https://hyper.ai/news/28189)**

* **中文解读：** [https://hyper.ai/news/28189](https://hyper.ai/news/28189)

* **科研团队：** 华东理工大学的李洪林课题组

* **相关研究：** ZINC 数据集、ChEMBL 数据库、深度学习模型、Transformer 架构、Macformer
* **发布期刊：** Nature Communication, 2023.07

* **论文链接：** [Macrocyclization of linear molecules by deep learning to facilitate macrocyclic drug candidates discovery](https://www.nature.com/articles/s41467-023-40219-8)

### **9. [快速自动扫描套件 FAST 高效获取样本信息](https://hyper.ai/news/28100)**

* **中文解读：** [https://hyper.ai/news/28100](https://hyper.ai/news/28100)

* **科研团队：** 美国阿贡国家实验室的研究团队

* **相关研究：** SLADS-Net 方法、路径优化技术。优先识别异质性区域、准确复制全扫描图像中所有主要特征

* **发布期刊：** Nature Communications, 2023.09

* **论文链接：** [Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy](https://www.nature.com/articles/s41467-023-40339-1)

### **10. [回归网络 + CGMD，预测百亿种多肽的自组装特性](https://hyper.ai/news/26408)**

* **中文解读：** [https://hyper.ai/news/26408](https://hyper.ai/news/26408)

* **科研团队：** 西湖大学的李文彬课题组

* **相关研究：** 拉丁超立方采样、CGMD 模型、AP 预测模型、Transformer、MLP、TRN 模型。得到了五肽和十肽的 AP

* **发布期刊：** Advanced Science, 2023.09

* **论文链接：** [Deep Learning Empowers the Discovery of Self-Assembling Peptides with Over 10 Trillion Sequences](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301544)

### **11. [无监督学习预测 7100 万种基因突变](https://hyper.ai/news/26154)**

* **中文解读：** [https://hyper.ai/news/26154](https://hyper.ai/news/26154)

* **科研团队：** 谷歌DeepMind 研究团队

* **相关研究：** ClinVar 数据集、AlphaFold、弱标签学习、无监督学习、AlphaMissense

* **发布期刊：** Science, 2023.09

* **论文链接：** [Accurate proteome-wide missense variant effect prediction with AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492)

### **12. [基于图神经网络 (GNN) 开发气味分析 AI](https://hyper.ai/news/25952)**

* **中文解读：** [https://hyper.ai/news/25952](https://hyper.ai/news/25952)

* **科研团队：** Google Research 的分支 Osmo 公司

* **相关研究：** GS-LF 数据库、GNN、贝叶斯优化算法。在 53% 的化学分子、55% 的气味描述词判断中优于人类

* **发布期刊：** Science, 2023.08

* **论文链接：** [A principal odor map unifies diverse tasks in olfactory perception](https://www.science.org/doi/full/10.1126/science.ade4401)

### **13. [图神经网络筛选安全高效的抗衰老成分](https://hyper.ai/news/25822)**

* **中文解读：** [https://hyper.ai/news/25822](https://hyper.ai/news/25822)

* **科研团队：** 麻省理工学院的研究团队

* **相关研究：** 深度学习、GNN、卷积神经网络。Chemprop 模型的正预测率为 11.6%，高于人工筛选的 1.9%

* **发布期刊：** Nature Aging, 2023.05

* **论文链接：** [Discovering small-molecule senolytics with deep neural networks](https://www.nature.com/articles/s43587-023-00415-z)

### **14. [机器学习量化分析多巴胺的释放量和释放位置](https://hyper.ai/news/25153)**

* **中文解读：** [https://hyper.ai/news/25153](https://hyper.ai/news/25153)

* **科研团队：** 美国加利福尼亚大学伯克利分校的研究团队

* **相关研究：** SVM、RF、机器学习。对刺激强度的判断准确率达 0.832、对多巴胺释放脑区的判断准确率达 0.708

* **发布期刊：** ACS Chemical Neuroscience, 2023.06

* **论文链接：** [Identifying Neural Signatures of Dopamine Signaling with Machine Learning](https://pubs.acs.org/doi/full/10.1021/acschemneuro.3c00001)

### **15. [机器学习发现三种抗衰老药物](https://hyper.ai/news/24578)**

* **中文解读：** [https://hyper.ai/news/24578](https://hyper.ai/news/24578)

* **科研团队：** 梅奥诊所的 James L. Kirkland 博士等人

* **相关研究：** 机器学习、随机森林模型、5倍交叉验证、随机森林（RF）模型。发现抗衰老药物 Ginkgetin、Periplocin 和 Oleandrin

* **发布期刊：** Nature Communications, 2023.06

* **论文链接：** [Discovery of Senolytics using machine learning](https://www.nature.com/articles/s41467-023-39120-1)

### **16. [深度学习筛选抑制鲍曼不动杆菌的新型抗生素](https://hyper.ai/news/24499)**

* **中文解读：** [https://hyper.ai/news/24499](https://hyper.ai/news/24499)

* **科研团队：** 麦克马斯特大学、麻省理工学院的研究团队

* **相关研究：** Broad 研究所的高通量筛选子库、机器学习、深度学习。筛选了大约 7,500 个分子，发现了一种名为 abaucin 的抗菌化合物

* **发布期刊：** Nature Chemical Biology, 2023.05

* **论文链接：** [Deep learning-guided discovery of an antibiotic targeting Acinetobacter baumannii](https://www.nature.com/articles/s41589-023-01349-8#access-options)

### **17. [机器学习模型应用于预测生物墨水可打印性](https://hyper.ai/news/24237)**

* **中文解读：** [https://hyper.ai/news/24237](https://hyper.ai/news/24237)

* **科研团队：** 圣地亚哥德孔波斯特拉大学、伦敦大学学院的研究团队

* **相关研究：** 机器学习模型、ANN、SVM、RF、kappa、R²、MAE。准确率高达 97.22%

* **发布期刊：** International Journal of Pharmaceutics: X, 2023.12

* **论文链接：** [Predicting pharmaceutical inkjet printing outcomes using machine learning](https://www.sciencedirect.com/science/article/pii/S2590156723000257)

### **18. [机器学习分化多能干细胞](https://hyper.ai/news/23940)**

* **中文解读：** [https://hyper.ai/news/23940](https://hyper.ai/news/23940)

* **科研团队：** 北京大学赵扬课题组、张钰课题组联合北京交通大学刘一研课题组

* **相关研究：** 活细胞成像技术、机器学习、弱监督模型、pix2pix 深度学习模型。分化效率从 21.6% ± 2.7% 提升至 88.8% ± 10.5%

* **发布期刊：** Cell Discovery, 2023.06

* **论文链接：** [A live-cell image-based machine learning strategy for reducing variability in PSC differentiation systems](https://www.nature.com/articles/s41421-023-00543-1)

### **19. [机器学习模型预测长效注射剂药物释放速率](https://hyper.ai/news/33892)**
* **中文解读：** [https://hyper.ai/news/33892](https://hyper.ai/news/33892)

* **科研团队：** 多伦多大学研究团队

* **相关研究：** MLR、Lasso、PLS、DT、RF、LGBM、XGB、自NGB、SVR、k-NN、NN、嵌套交叉验证、最远邻聚类算法

* **发布期刊：** Nature Communications, 2023.01

* **论文链接：** [Machine learning models to accelerate the design of polymeric long-acting injectables](https://www.nature.com/articles/s41467-022-35343-w)


### **20. [机器学习算法有效预测植物抗疟性](https://hyper.ai/news/33883)**

* **中文解读：** [https://hyper.ai/news/33883](https://hyper.ai/news/33883)

* **科研团队：** 英国皇家植物园及圣安德鲁斯大学的研究团队

* **相关研究：** Logit、SVC、XGB、BNN、GridSearchCV 算法、10 折分层交叉验证、马尔可夫链蒙特卡洛迭代。准确率为 0.67

* **发布期刊：** Frontiers in Plant Science, 2023.05

* **论文链接：** [Machine learning enhances prediction of plants as potential sources of antimalarials](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10248027)

### **21. [机器学习集成方法预测病毒蛋白片段免疫原性](https://hyper.ai/news/30786)**
* **中文解读：** [https://hyper.ai/news/30786](https://hyper.ai/news/30786)

* **科研团队：** 北京航空航天大学李静研究团队

* **相关研究：** 蛋白质数据库 UniProt、Protegen 数据库、集成机器学习方法 VirusImmu、RF 、 XGBoost 、kNN、随机采样交叉验证

* **发布期刊：** bioRxiv, 2023.11

* **论文链接：** [VirusImmu: a novel ensemble machine learning approach for viral immunogenicity prediction](https://www.biorxiv.org/content/10.1101/2023.11.23.568426v1)

### **22. [用生成式 AI 开发新型抗生素](https://hyper.ai/news/31421)**
* **中文解读：** [https://hyper.ai/news/31421](https://hyper.ai/news/31421)

* **科研团队：** 麦马、斯坦福团队

* **相关研究：** Pharmakon-1760 库、药物再利用中心数据库、合成小分子筛选集、蒙特卡洛树搜索 、生成式人工智能模型 SyntheMol、生成 24,335 个完整分子、设计出易于合成的新型化合物

* **发布期刊：** Nature Machine Intelligence, 2024.03

* **论文链接：** [Generative AI for designing and validating easily synthesizable and structurally novel antibiotics](https://www.nature.com/articles/s42256-024-00809-7 )

### **23. [基于深度学习研发一种自动化、高速、多维的单粒子追踪系统](https://hyper.ai/news/31341)**
* **中文解读：** [https://hyper.ai/news/31341](https://hyper.ai/news/31341)

* **科研团队：** 厦门大学方宁教授团队

* **相关研究：** 多维成像设备、双焦平面成像、视差显微镜、多维成像设备、卷积神经网络模型、抗噪性和鲁棒性

* **发布期刊：** Nature Machine Intelligence, 2024.03

* **论文链接：** [Deep Learning-Assisted Automated Multidimensional Single Particle Tracking in Living Cells](https://doi.org/10.1021/acs.nanolett.3c04870)

### **24. [ProEnsemble 机器学习框架：优化进化通路启动子组合](https://hyper.ai/news/30594)**
* **中文解读：** [https://hyper.ai/news/30594](https://hyper.ai/news/30594)

* **科研团队：** 中科院罗小舟团队

* **相关研究：** 合成生物、基因上位效应、自动化平台、十折交叉验证、集成模型、Gradient Boosting Regressor、Ridge Regressor、Gradient Boosting、通用型底盘高效合成黄酮类化合物

* **发布期刊：** ADVANCED SCIENCE, 2024.02

* **论文链接：** [Pathway Evolution Through a Bottlenecking-Debottlenecking Strategy and Machine Learning-Aided Flux Balancing](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202306935)

### **25. [微环境感知图神经网络 ProtLGN 指导蛋白质定向进化](https://hyper.ai/news/32246)**
* **中文解读：** [https://hyper.ai/news/32246](https://hyper.ai/news/32246)

* **科研团队：** 上海交通大学洪亮课题组

* **相关研究：** 微环境感知图神经网络、轻量级图神经去噪网络、自监督预训练、等变图神经网络、超过 40% 的 PROTLGN 设计单点突变体蛋白质优于其野生型对应物

* **发布期刊：** JOURNAL OF CHEMICAL INFORMATION AND MODELING, 2024.04

* **论文链接：** [Protein Engineering with Lightweight Graph Denoising Neural Networks](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00036)

### **26. [深度学习模型 AlphaPPIMd：用于蛋白质-蛋白质复合物构象集合探索](https://hyper.ai/news/32435)**
* **中文解读：** [https://hyper.ai/news/32435](https://hyper.ai/news/32435)

* **科研团队：** 延世大学王建民团队

* **相关研究：** 深度学习、生成式 AI、Transformer、生成神经网络学习、分子动力学、barnase-barstar 复合物轨迹集、蛋白质数据库 Protein Data Bank、AlphaPPIMd 模型、自注意力机制、特征优化模块、注意力分数、全原子模型、模型的平均训练精度为 0.995、平均验证精度为 0.999 

* **发布期刊：** Journal of Chemical Theory and Computation, 2024.05

* **论文链接：** [Exploring the conformational ensembles of protein-protein complex with transformer-based generative model](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00255)

### **27. [新型肿瘤抑制蛋白降解剂 dp53m 可抑制癌细胞增殖](https://hyper.ai/news/32527)**
* **中文解读：** [https://hyper.ai/news/32527](https://hyper.ai/news/32527)

* **科研团队：** 西交利物浦大学慧湖药学院吴思晋教授、天津医科大学总医院谢松波教授、钟殿胜教授团队

* **相关研究：** MD 模拟、迭代分子对接引导 post-SELEX 法、dp53m 可特异性识别 p53-R175H 蛋白，并对其进行降解

* **发布期刊：** Science Bulletin, 2024.05

* **论文链接：** [An engineered DNA aptamer-based PROTAC for precise therapy of p53-R175H hotspot mutant-driven cancer](https://www.sciencedirect.com/science/article/pii/S2095927324003517)

### **28. [CVPR 最佳学生论文！多模态模型 BioCLIP 实现零样本学习](https://hyper.ai/news/32544)**
* **中文解读：** [https://hyper.ai/news/32544](https://hyper.ai/news/32544)

* **科研团队：** 俄亥俄州立大学 Jiaman Wu 团队

* **相关研究：** 生物图像数据集 TreeOfLife-10M、多模态模型、计算机视觉、视觉编码器、文本编码器、自回归语言模型、模型在零样本和少样本任务中均表现出色

* **发布期刊：** CVPR 2024, 2024.02

* **论文链接：** [BIoCLIP: A Vision Foundation Model for the Tree of Life](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html)

### **29. [1 亿参数！细胞大模型 scFoundation 可对 2 万基因同时建模](https://hyper.ai/news/32623)**
* **中文解读：** [https://hyper.ai/news/32623](https://hyper.ai/news/32623)

* **科研团队：** 清华大学自动化系生命基础模型实验室主任张学工教授、电子系/AIR 马剑竹教授和百图生科宋乐博士

* **相关研究：** 人工智能细胞大模型、人类单细胞组学数据 DISCO，欧洲分子生物学实验室-欧洲生物信息学研究所数据库 EMBL-EBI、GEO 数据集，Single Cell Portal 数据集，HCA 数据集，hECA 数据集、Transformer、非对称的编码器-解码器结构、向量模块、RDA 建模

* **发布期刊：** Nature Methods, 2024.06

* **论文链接：** [Large-scale foundation model on single-cell transcriptomics](https://www.nature.com/articles/s41592-024-02305-7)

### **30. [入选顶会 ICML，蛋白质语言模型 ESM-AA 超越传统 SOTA](https://hyper.ai/news/32674)**

* **中文解读：** [https://hyper.ai/news/32674](https://hyper.ai/news/32674)

* **科研团队：** 清华大学周浩教授联合北京大学、南京大学和水木分子团队

* **相关研究：** 蛋白质数据集 AlphaFold DB、蛋白质数据集 Dp 和一个分子数据集 Dm、解压缩、多尺度掩码语言建模

* **发布期刊：** ICML 2024, 2024.06

* **论文链接：** [ESM All-Atom: Multi-scale Protein Language Model for Unified Molecular Modeling](https://icml.cc/virtual/2024/poster/35119)

### **31. [SPACE 算法登 Cell 子刊！组织模块发现能力领先同类工具](https://hyper.ai/news/32738)**
* **中文解读：** [https://hyper.ai/news/32738](https://hyper.ai/news/32738)

* **科研团队：** 清华大学张强锋课题组

* **相关研究：** 空间转录组学、STARmap 小鼠 PLA 数据集、MERFISH 小鼠 AB 数据集、MERFISH 小鼠 WB 数据集、Xenium 人类 BC 数据集、CosMx 人类 NSCLC 数据集、Visium 人脑数据集、编码器、邻近图解码器、基因表达解码器、空间邻近性、自监督学习

* **发布期刊：** Cell Systems, 2024.06

* **论文链接：** [Tissue module discovery in single-cell resolution spatial transcriptomics data via cell-cell interaction-aware cell embedding](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00124-8)

### **32. [基于 AlphaFold 实现新突破，揭示蛋白质动态多样性](https://hyper.ai/news/33075)**
* **中文解读：** [https://hyper.ai/news/33075](https://hyper.ai/news/33075)

* **科研团队：** 麻省理工学院研究团队

* **相关研究：** 流匹配技术、蛋白质语言模型、神经网络、AlphaFold、ESMFold

* **发布期刊：** ICML 2024, 2024.06

* **论文链接：** [AlphaFold Meets Flow Matching for Generating Protein Ensembles](https://openreview.net/forum?id=rs8Sh2UASt)

### **33. [基于扩散模型开发 P450 酶从头设计方法 P450Diffusion](https://hyper.ai/news/33057)**

* **中文解读：** [https://hyper.ai/news/33057](https://hyper.ai/news/33057)

* **科研团队：** 中国科学院天津工业生物技术研究所江会锋、程健团队

* **相关研究：** 定向进化、扩散模型、深度学习、去噪扩散概率模型、三点固定、微调扩散模型 、预训练、催化能力提高 3.5 倍

* **发布期刊：** Research, 2024.07

* **论文链接：** [Cytochrome P450 Enzyme Design by Constraining the Catalytic Pocket in a Diffusion Model](https://spj.science.org/doi/10.34133/research.0413)

### **34. [将等变图神经网络用于靶蛋白结合位点预测，性能提升 20%](https://hyper.ai/news/32957)**
* **中文解读：** [https://hyper.ai/news/32957](https://hyper.ai/news/32957)

* **科研团队：** 中国人民大学高瓴人工智能学院的研究团队

* **相关研究：** E(3) 等变图神经网络、卷积神经网络、EquiPocket 框架、scPDB 数据集、PDBbind 数据集、COACH 420 数据集、HOLO4K 数据集、局部几何建模模块、全局结构建模模块 、表面信息传递模块

* **发布期刊：** ICML 2024, 2024.07

* **论文链接：** [EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction](https://openreview.net/forum?id=1vGN3CSxVs)

### **35. [20 个实验数据创造 AI 蛋白质里程碑！FSFP 有效优化蛋白质预训练模型](https://hyper.ai/news/32822)**
* **中文解读：** [https://hyper.ai/news/32822](https://hyper.ai/news/32822)

* **科研团队：** 上海交通大学自然科学研究院/物理天文学院/张江高研院/药学院洪亮教授课题组，联合上海人工智能实验室青年研究员谈攀团队

* **相关研究：** 蛋白质突变数据集 ProteinGym、预训练蛋白质语言模型、元迁移学习、排序学习、参数高效微调、LTR 技术、有效优化蛋白质语言模型的训练策略 FSFP、模型无关元学习方法

* **发布期刊：** Nature Communications, 2024.07

* **论文链接：** [Enhancing efficiency of protein language models with minimal wet-lab data through few-shot learning](https://doi.org/10.1038/s41467-024-49798-6)

### **36. [可迁移深度学习模型鉴定多类型 RNA 修饰、显著减少计算成本](https://hyper.ai/news/32745)**
* **中文解读：** [https://hyper.ai/news/32745](https://hyper.ai/news/32745)

* **科研团队：** 上海交通大学生命科学技术学院长聘教轨副教授余祥课题组，联合上海辰山植物园杨俊 / 王红霞团队

* **相关研究：** 可迁移深度学习模型 TandemMod、体外转录数据集 ELIGOS、Curlcake 数据集、体外表观转录组数据集 IVET、一维卷积神经网络、双向长短期记忆模块、注意力机制、全连接层 (full-connected layers) 的分类器

* **发布期刊：** Nature Communications, 2024.05

* **论文链接：** [Transfer learning enables identification of multiple types of RNA modifications using nanopore direct RNA sequencing](https://www.nature.com/articles/s41467-024-48437-4)

### **37. [InstructProtein：利用知识指令对齐蛋白质语言与人类语言](https://hyper.ai/news/33697)**
* **中文解读：** [https://hyper.ai/news/33697](https://hyper.ai/news/33697)

* **科研团队：** 浙江大学陈华钧、张强团队

* **相关研究：** 大语言模型、蛋白质知识指令数据集、Gene Ontology (GO) 数据集、InstructProtein、知识图谱、蛋白质位置预测、蛋白质功能预测 、蛋白质金属离子结合能力预测

* **发布期刊：** ACL 2024, 2023.10

* **论文链接：** [InstructProtein: Aligning Human and Protein Language via Knowledge Instruction](https://arxiv.org/abs/2310.03269)

### **38. [蛋白质-文本生成框架 ProtT3 实现蛋白质数据与文本信息跨模态解读](https://hyper.ai/news/33546)**
* **中文解读：** [https://hyper.ai/news/33546](https://hyper.ai/news/33546)

* **科研团队：** 中国科学技术大学王翔，联合新加坡国立大学刘致远团队、北海道大学研究团队

* **相关研究：** 跨模态投影器、蛋白质语言模型、Swiss-Prot 和 ProteinKG25 数据集、PDB-QA 数据集

* **发布期刊：** ACL 2024, 2023.05

* **论文链接：** [ProtT3: Protein-to-Text Generation for Text-based Protein Understanding](https://arxiv.org/abs/2405.12564)

### **39. [CPDiffusion 模型，超低成本、全自动设计功能型蛋白质](https://hyper.ai/news/34692)**
* **中文解读：** [https://hyper.ai/news/34692](https://hyper.ai/news/34692)

* **科研团队：** 上海交通大学自然科学研究院、物理与天文学院、张江高等研究院、药学院洪亮课题组

* **相关研究：** 蛋白质工程、扩散概率模型框架 CPDiffusion、氨基酸、图神经网络、辅助药物设计、蛋白质语言模型、 CATH 4.2 数据集

* **发布期刊：** Cell Discovery,  2024.09

* **论文链接：** [A conditional protein diffusion model generates artificial programmable endonuclease sequences with enhanced activity](https://www.nature.com/articles/s41421-024-00728-2)
### **40. [基于蛋白质语言模型和密集检索技术，一种全新的蛋白质同源物检测方法](https://hyper.ai/news/34225)**
* **中文解读：** [https://hyper.ai/news/34225](https://hyper.ai/news/34225)

* **科研团队：** 香港中文大学李煜、复旦大学智能复杂体系实验室、上海人工智能实验室青年研究员孙思琦、耶鲁大学 Mark Gerstein 

* **相关研究：** 蛋白质工程、蛋白质语言模型、密集检索技术、密集同源物检索器 、混合模型 DHR-meta、UR90 数据集、JackHMMER 算法、BFD/MGnify 数据集、DHR 方法、蛋白质同源物检测灵敏度提高 56%

* **发布期刊：** Nature Biotechnology, 2024.08

* **论文链接：** [Fast, sensitive detection of protein homologs using deep dense retrieval](https://doi.org/10.1038/s41587-024-02353-6)
### **41. [AlphaProteo 可高效设计靶蛋白结合物，亲和力提高 300 倍](https://hyper.ai/news/34214)**
* **中文解读：** [https://hyper.ai/news/34214](https://hyper.ai/news/34214)

* **科研团队：** DeepMind、弗朗西斯·克里克研究所

* **相关研究：** 蛋白质工程、蛋白质语言模型、AI 药物设计、靶蛋白 、AI 工具、机器学习模型 AlphaProteo、VEGF-A 蛋白结合体设计、生成模型 (Generator) 、过滤器 (Filter)、候选结合物与靶蛋白结合数量高出 5-100 倍

* **发布期刊：** DeepMind, 2024.09

* **论文链接：** [AlphaProteo 为生物学和健康研究生成新型蛋白质](https://deepmind.google/discover/blog/alphaproteo-generates-novel-proteins-for-biology-and-health-research/) 
### **42. [全新去噪蛋白质语言模型 DePLM，突变效应预测优于 SOTA 模型](https://hyper.ai/cn/news/34954)**
* **中文解读：** [https://hyper.ai/cn/news/34954](https://hyper.ai/cn/news/34954)

* **科研团队：** 浙江大学计算机科学与技术学院、浙江大学国际联合学院、浙江大学杭州国际科创中心陈华钧教授、张强博士

* **相关研究：** 去噪蛋白质语言模型 (DePLM)、ProteinGym 深度突变筛选 (DMS) 实验集合、DMS 数据集、随机交叉验证方法、泛化能力实验、基于排序信息的前向过程来扩展扩散模型以去噪进化信息、基于排序的去噪扩散过程、排序算法 (sorting algorithm) 生成轨迹、PromptProtein 模型

* **发布期刊：** NeurIPS 2024

* **论文链接：** [DePLM: Denoising Protein Language Models for Property Optimization](https://neurips.cc/virtual/2024/poster/95517 ) 
### **43                                                      . [几何深度生成模型 DynamicBind，实现蛋白质动态对接预测](https://hyper.ai/cn/news/34894)**
* **中文解读：** [https://hyper.ai/cn/news/34894](https://hyper.ai/cn/news/34894)

* **科研团队：** 上海交通大学郑双佳课题组、星药科技、中山大学药学院、美国莱斯大学

* **相关研究：**  PDBbind 数据集、MDT 测试集、深度扩散模型、等变几何神经网络技术、PDB 格式的类结构、小分子配体格式、contact-LDDT (cLDDT) 评分模块、AlphaFold 结构、亲和力预测模块、生成式人工智能技术

* **发布期刊：** Nature Communications, 2024.2

* **论文链接：** [DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model](https://www.nature.com/articles/s41467-024-45461-2)
## **AI+   医疗健康：AI+Healthcare**

### **1. [深度学习系统 DeepDR Plus 用眼底图像预测糖尿病视网膜病变](https://hyper.ai/news/29769)**
* **中文解读：** [https://hyper.ai/news/29769](https://hyper.ai/news/29769)

* **科研团队：** 上海交通大学贾伟平、李华婷和盛斌教授团队，清华大学黄天荫研究团队

* **相关研究：** SDPP 数据、DRPS 数据、ResNet-50、眼底模型、自监督学习、 IBS 评估模型、元数据模型、组合模型。将临床应用的平均筛查间隔从 12 个月延长至 31.97 个月

* **发布期刊：** Nature Medicine, 2024.01

* **论文链接：** [A deep learning system for predicting time to progression of diabetic retinopathy](https://www.nature.com/articles/s41591-023-02702-z)

### **2. [逻辑回归模型分析高绿色景观指数可降低 MetS 风险](https://hyper.ai/news/29559)**
* **中文解读：** [https://hyper.ai/news/29559](https://hyper.ai/news/29559)

* **科研团队：** 浙江大学吴息凤研究团队

* **相关研究：** 卷积神经网络模型、逻辑回归模型、Isochrone API

* **发布期刊：** Environment International, 2024.01

* **论文链接：** [Beneficial associations between outdoor visible greenness at the workplace and metabolic syndrome in Chinese adults](https://doi.org/10.1016/j.envint.2023.108327)

### **3. [深度学习系统助力初级眼科医生的诊断一致性提高 12%](https://hyper.ai/news/29549)**
* **中文解读：** [https://hyper.ai/news/29549](https://hyper.ai/news/29549)

* **科研团队：** 北京协和医院、四川大学华西医院、河北医科大学第二医院、天津医科大学眼科医院、温州医科大学附属眼视光医院、北京致远慧图科技有限公司、中国人民大学研究团队

* **相关研究：** quality assessment model、diagnostic model、CNN。为 13 种眼底疾病的自动检测提供新方法

* **发布期刊：** npj digital medicine, 2024.01

* **论文链接：** [The performance of a deep learning system in assisting junior ophthalmologists in diagnosing 13 major fundus diseases: a prospective multi-center clinical trial](https://www.nature.com/articles/s41746-023-00991-9)

### **4. [GSP-GCNs 实现帕金森病诊断准确率高达 90.2%](https://hyper.ai/news/29189)**
* **中文解读：** [https://hyper.ai/news/29189](https://hyper.ai/news/29189)

* **科研团队：** 中科院深圳先进技术研究院和中山大学附属第一医院研究团队

* **相关研究：** 图信号处理模块 (GSP) 、图网络模块 (graph-network module) 、分类器 (classifier) 、可解释模型 ( interpretable model)

* **发布期刊：** npj Digital Medicine, 2024.01

* **论文链接：** [An interpretable model based on graph learning for diagnosis of Parkinson’s disease with voice-related EEG](https://www.nature.com/articles/s41746-023-00983-9)

### **5. [乳腺癌预后评分系统 MIRS](https://hyper.ai/news/29304)**
* **中文解读：** [https://hyper.ai/news/29304](https://hyper.ai/news/29304)

* **科研团队：** 美国肯塔基大学、澳门科技大学、澳门大学、广州医科大学研究团队

* **相关研究：** TCGA 数据库、神经网络模型、预后评分系统、ESTIMATE 算法、机器学习、XGboost 、 Borota RF、ElasticNet

* **发布期刊：** iScience, 2023.11

* **论文链接：** [MIRS: An AI scoring system for predicting the prognosis and therapy of breast cancer](https://doi.org/10.1016/j.isci.2023.108322)

### **6. [视网膜图像基础模型 RETFound，预测多种系统性疾病](https://hyper.ai/news/28113)**
* **中文解读：** [https://hyper.ai/news/28113](https://hyper.ai/news/28113)

* **科研团队：** 伦敦大学学院和 Moorfields 眼科医院的在读博士周玉昆等人

* **相关研究：** 自监督学习、MEH-MIDAS 数据集、EyePACS 数据集、SL-ImageNet、SSL-ImageNet、SSL-Retinal。RETFound 模型预测 4 种疾病的性能均超越对比模型

* **发布期刊：** Nature, 2023.08

* **论文链接：** [A foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x)

### **7. [SVM 优化触觉传感器，盲文识别率达 96.12%](https://hyper.ai/news/26561)**
* **中文解读：** [https://hyper.ai/news/26561](https://hyper.ai/news/26561)

* **科研团队：** 浙江大学的杨赓和徐凯臣课题组

* **相关研究：** SVM算法、机器学习、CNN、自适应矩估计算法。优化后的传感器能准确识别 6 种动态触摸模式

* **发布期刊：** Advanced Science, 2023.09

* **论文链接：** [Machine Learning-Enabled Tactile Sensor Design for Dynamic Touch Decoding](https://onlinelibrary.wiley.com/doi/10.1002/advs.202303949)

### **8. [中科院基因组所建立开放生物医学成像档案](https://hyper.ai/news/26334)**
* **中文解读：** [https://hyper.ai/news/26334](https://hyper.ai/news/26334)

* **科研团队：** 中科院基因组所

* **相关研究：** TCIA 癌症影像数据库、de-identification、quality control、Collection、Individual、Study、 Series, Image、三元组网络、attention module

* **发布期刊：** bioRxiv, 2023.08

* **论文链接：** [Self-supervised learning of hologram reconstruction using physics consistency](https://www.nature.com/articles/s42256-023-00704-7)

### **9. [AI Lunit 阅读乳腺 X 光片的准确率与医生相当](https://hyper.ai/news/26135)**
* **中文解读：** [https://hyper.ai/news/26135](https://hyper.ai/news/26135)

* **科研团队：** 英国诺丁汉大学的研究团队

* **相关研究：** PERFORMS 数据集，标注 + 评分。AI 的灵敏度与医生一致、特异性与医生没有显著差异

* **发布期刊：** Radiology, 2023.09

* **论文链接：** [Performance of a Breast Cancer Detection AI Algorithm Using the Personal Performance in Mammographic Screening Scheme](https://pubs.rsna.org/doi/10.1148/radiol.223299)

### **10. [特征选择策略检测乳腺癌生物标志物](https://hyper.ai/news/24589)**
* **中文解读：** [https://hyper.ai/news/24589](https://hyper.ai/news/24589)

* **科研团队：** 意大利那不勒斯费德里科二世大学研究团队

* **相关研究：** 机器学习、特征选择策略、TCGA/GEO 数据集、Gain Ratio、RF、SVM-RFE。SVM-RFE 的稳定性和获得的 signature 预测能力最高

* **发布期刊：** CIBB 2023, 2023.07

* **论文链接：** [Robust Feature Selection strategy detects a panel of microRNAs as putative diagnostic biomarkers in Breast Cancer](https://www.researchgate.net/publication/372083934)

### **11. [梯度提升机模型准确预测 BPSD 亚综合征](https://hyper.ai/news/23926)**
* **中文解读：** [https://hyper.ai/news/23926](https://hyper.ai/news/23926)

* **科研团队：** 韩国延世大学研究团队

* **相关研究：** 机器学习模型、多重插补方法、逻辑回归模型、随机森林模型、梯度提升机模型、支持向量机模型。梯度提升机模型平均 AUC 值最高

* **发布期刊：** Scientifc Reports, 2023.05

* **论文链接：** [Machine learning‑based predictive models for the occurrence of behavioral and psychological symptoms of dementia: model development and validation](https://www.nature.com/articles/s41598-023-35194-5)

### **12. [机器学习模型预测患者一年内死亡率](https://hyper.ai/news/33905)**
* **中文解读：** [https://hyper.ai/news/33905](https://hyper.ai/news/33905)

* **科研团队：** 中国湖北省麻城市人民医院的研究团队

* **相关研究：** 逻辑回归模型、机器学习模型、GBM、RF、DT。良好的临床实用性，与一年死亡率相关的前 3 个特征分别是 NT-proBNP、白蛋白和他汀类药物

* **发布期刊：** Cardiovascular Diabetology, 2023.06

* **论文链接：** [Machine learning-based models to predict one-year mortality among Chinese older patients with coronary artery disease combined with impaired glucose tolerance or diabetes mellitus](https://cardiab.biomedcentral.com/articles/10.1186/s12933-023-01854-z)

### **13. [AI 新脑机技术让失语患者「开口说话」](https://hyper.ai/news/33914)**

* **中文解读：** [https://hyper.ai/news/33914](https://hyper.ai/news/33914)

* **科研团队：** 加州大学团队

* **相关研究：** nltk Twitter 语料库、多模态语音神经假体、脑机接口、深度学习模型、Cornell 电影语料库、合成语音算法、机器学习

* **发布期刊：** Nature, 2023.08

* **论文链接：** [A high-performance neuroprosthesis for speech decoding and avatar control](https://www.nature.com/articles/s41586-023-06443-4)

### **14. [基于深度学习的胰腺癌人工智能检测](https://hyper.ai/news/33923)**
* **中文解读：** [https://hyper.ai/news/33923](https://hyper.ai/news/33923)

* **科研团队：** 阿里达摩院联合多家国内外医疗机构

* **相关研究：** 深度学习、PANDA、nnU-Net、CNN、Transformer。PANDA 检测到了 5 例癌症和 26 例临床漏诊病例

* **发布期刊：** Nature Medicine, 2023.11

* **论文链接：** [Large-scale pancreatic cancer detection via non-contrast CT and deep learning](https://www.nature.com/articles/s41591-023-02640-w)

### **15. [机器学习辅助肺癌筛查的群体有效性](https://hyper.ai/news/31197)**
* **中文解读：** [https://hyper.ai/news/31197](https://hyper.ai/news/31197)

* **科研团队：** 谷歌研究中心

* **相关研究：** DS_CA 数据集、DS_NLST 数据集、DS_US 数据集、DS_JPN 数据集、机器学习模型、肺癌筛查。特异性提高 5%-7%、病例筛查时间减少 14 秒

* **发布期刊：** Radiology AI, 2024.03

* **论文链接：** [Assistive AI in Lung Cancer Screening: A Retrospective Multinational Study in the United States and Japan](https://pubs.rsna.org/doi/10.1148/ryai.230079)


### **16. [卵巢癌诊断人工智能融合模型 MCF，输入常规实验室检验数据和年龄即可计算卵巢癌的患病风险](https://hyper.ai/news/30730)**
* **中文解读：** [https://hyper.ai/news/30730](https://hyper.ai/news/30730)

* **科研团队：** 中山大学刘继红研究团队

* **相关研究：** 特征选择方法、机器学习分类器、五倍交叉验证、多准则决策理论、融合 20 个基础分类模型、识别卵巢癌的准确率优于 CA125 和 HE4  等传统生物标志物

* **发布期刊：** ARTICLES, 2024.03

* **论文链接：** [Artificial intelligence-based models enabling accurate diagnosis ofovarian cancer using laboratory tests in China: a multicentre,retrospective cohort study](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00245-5/fulltext)

### **17. [谷歌发布 HEAL 架构，4 步评估医学 AI 工具是否公平](https://hyper.ai/news/31535)**
* **中文解读：** [https://hyper.ai/news/31535](https://hyper.ai/news/31535)

* **科研团队：** Google 研究团队

* **相关研究：** 机器学习、HEAL (The health equity framework) 框架、逻辑回归分析、交叉性分析、健康公平

### **18. [借鉴语义分割，开发空间转录组语义注释工具 Pianno](https://hyper.ai/news/31573)**
* **中文解读：** [https://hyper.ai/news/31573](https://hyper.ai/news/31573)

* **科研团队：** 复旦大学脑科学研究院诸颖团队

* **相关研究：** 计算机视觉、机器学习、空间聚类方法、无监督聚类方法、空间泊松点过程 (spatial Poisson point process, sPPP) 模型、高阶马尔科夫随机场 (Markov random field, MRF) 先验模型

* **发布期刊：** Nature Communications, 2024.04

* **论文链接：** [Pianno: a probabilistic framework automating semantic annotation for spatial transcriptomics](https://www.nature.com/articles/s41467-024-47152-4)

### **19. [AI 模型 UniFMIR，突破现有荧光显微成像极限](https://hyper.ai/news/31885)**
* **中文解读：** [https://hyper.ai/news/31885](https://hyper.ai/news/31885)

* **科研团队：** 复旦大学计算机科学技术学院颜波团队

* **相关研究：** UniFMIR 模型、多头模块、特征增强模块、多尾模块、Swin Transformer、自适应矩估计、深度学习、SR 模型、单图像超分辨率模型、U-Net

* **发布期刊：**Nature Methods, 2024.04

* **论文链接：** [Pretraining a foundation model for generalizable fluorescence microscopy-based image restoration](https://www.nature.com/articles/s41592-024-02244-3)

### **20. [深度学习系统，提高癌症生存预测准确性](https://hyper.ai/news/32068)**
* **中文解读：** [https://hyper.ai/news/32068](https://hyper.ai/news/32068)

* **科研团队：** 上海国家应用数学中心（上海交通大学分中心）俞章盛课题组（生命科学技术学院/医学院临床研究中心）

* **相关研究：** 深度学习系统、ST 数据集、integrated graph 和图深度学习的模型、卷积神经网络和图神经网络、外部测试集 MCO-CRC、空间基因表达预测模型、super-patch graph 生存模型、H&E 染色组织学图像 (H&E-stained histological image) 预处理、IGI-DL 模型

* **发布期刊：**Cell Reports Medicine, 2024.05

* **论文链接：** [Harnessing TME depicted by histological images to improve cancer prognosis through a deep learning system](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(24)00205-2 )

### **21. [MemSAM 将「分割一切」模型用于医学视频分割](https://hyper.ai/news/32372)**
* **中文解读：** [https://hyper.ai/news/32372](https://hyper.ai/news/32372)

* **科研团队：** 深圳大学吴惠思

* **相关研究：** 视觉模型、医学视频分割、超声心动图视频分割模型、记忆强化机制、超声心动图数据集 CAMUS  和 EchoNet-Dynamic、图像编码器、提示编码器、掩码解码器、Softmax 函数、基于 CNN 的 UNet 、基于 Transformer 的 SwinUNet、CNN-Transformer 混合的 H2Former、SonoSAM 模型、SAMUS 模型

* **发布期刊：** CVPR 2024, 2024.05

* **论文链接：** [MemSAM: Taming Segment Anything Model forEchocardiography Video Segmentation](https://github.com/dengxl0520/MemSAM)

### **22. [医学图像分割模型 Medical SAM 2 刷新医学图像分割 SOTA 榜](https://hyper.ai/news/33738)**
* **中文解读：** [https://hyper.ai/news/33738](https://hyper.ai/news/33738)

* **科研团队：** 牛津大学团队

* **相关研究：** 医学图像分割模型、SAM 2、SA-V 视频分割数据集、Medical SAM 2 示例医学分割数据集、 图像编码器、记忆编码器、记忆注意力机制

* **发布期刊：** arXiv, 2024.08

* **论文链接：** [Medical SAM 2: Segment medical images as video via Segment Anything Model 2](https://arxiv.org/abs/2408.00874)

### **23. [机器学习抗击化疗耐药性与肿瘤复发，构筑乳腺癌干细胞的有力防线](https://hyper.ai/news/33566)**
* **中文解读：** [https://hyper.ai/news/33566](https://hyper.ai/news/33566)

* **科研团队：** 山东大学吕海泉、孙蓉、张凯及山西医科大学梅齐，联合螺旋矩阵公司等研究团队

* **相关研究：** 机器学习、乳腺浸润性癌 (BRCA) 数据集、皮尔逊相关系数分析、基因集富集分析、评估乳腺癌患者样本中的癌症干细胞特征

* **发布期刊：** Advanced Science, 2024.07

* **论文链接：** [Polyamine Anabolism Promotes Chemotherapy-Induced Breast Cancer Stem Cell Enrichment](https://onlinelibrary.wiley.com/doi/10.1002/advs.202404853)

### **24. [糖尿病诊疗的视觉-大语言模型 DeepDR-LLM 登 Nature 子刊](https://hyper.ai/news/33292)**
* **中文解读：** [https://hyper.ai/news/33292](https://hyper.ai/news/33292)

* **科研团队：** 清华大学副教务长、医学院主任黄天荫教授团队，上海交通大学电院计算机系/教育部人工智能重点实验室盛斌教授团队，上海交通大学医学院附属第六人民医院贾伟平教授及李华婷教授团队，新加坡国立大学及新加坡国家眼科中心覃宇宗教授团队

* **相关研究：** 大语言模型、基于眼底图像的深度学习技术、融合适配器 (Adaptor) 和低秩自适应、Transformer 模型架构、监督微调方法、可提高基层 DR 筛查能力和糖尿病诊疗水平

* **发布期刊：** Nature Medicine, 2024.07

* **论文链接：** [Integrated image-based deep learning and language models for primary diabetes care](https://www.nature.com/articles/s41591-024-03139-8)

### **25. [水平直逼高级病理学家！清华团队提出 AI 基础模型 ROAM，实现胶质瘤精准诊断](https://hyper.ai/news/33136)**
* **中文解读：** [https://hyper.ai/news/33136](https://hyper.ai/news/33136)

* **科研团队：** 清华大学自动化系生命基础模型实验室闾海荣副研究员、江瑞教授、张学工教授与中南大学湘雅医院胡忠良教授团队

* **相关研究：** 基于大区域兴趣 (large regions of interest) 和金字塔 Transformer (pyramid transformer) 、精准病理诊断 AI 基础模型 ROAM、大尺寸图像块和多尺度特征学习模块、湘雅胶质瘤 WSI 数据集、TCGA 胶质瘤 WSI 数据集、弱监督计算病理学方法、卷积神经网络

* **发布期刊：** Nature Machine Intelligence, 2024.06

* **论文链接：** [A transformer-based weakly supervised computational pathology method for clinical-grade diagnosis and molecular marker discovery of gliomas](https://www.nature.com/articles/s42256-024-00868-w)

### **26. [医学图像分割通用模型 ScribblePrompt，性能优于 SAM](https://hyper.ai/news/34720)**
* **中文解读：** [https://hyper.ai/news/34720](https://hyper.ai/news/34720)

* **科研团队：** 美国麻省理工学院计算机科学与人工智能实验室团队、麻省总医院、哈佛医学院

* **相关研究：** 深度学习、医学图像分割、MegaMedical 数据集、交互式分割方法、生物医学成像数据集、生物医学图像分割的通用模型 ScribblePrompt、生成合成标签机制、全卷积架构、ScribblePrompt 架构、CNN-Transformer 混合解决方案
* **发布期刊：** ECCV 2024, 2024.07

* **论文链接：** [ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image](https://arxiv.org/pdf/2312.07381)
### **27. [数字孪生脑平台，展现出类似人脑中观测的临界现象与相似认知功能](https://hyper.ai/news/34573)**
* **中文解读：** [https://hyper.ai/news/34573](https://hyper.ai/news/34573)

* **科研团队：** 复旦大学类脑智能科学与技术研究院冯建峰教授团队

* **相关研究：**  神经元网络、数字孪生大脑、逆向工程技术、脑科学、全脑范围内的尖峰神经元网络、磁共振成像技术、快速梯度回波序列、cortico-subcortical 模型、DTB 模型、分析了神经元数量和平均突触连接度对模型与生物数据相似度的影响、同化模型

* **发布期刊：** National Science Review, 2024.5

* **论文链接：** [Imitating and exploring human brain’s resting and task-performing states via resembling brain computing: scaling and architecture](https://doi.org/10.1093/nsr/nwae080)
### **28. [自动化大模型对话 Agent 模拟系统，可初诊抑郁症](https://hyper.ai/cn/news/34845)**
* **中文解读：** [https://hyper.ai/cn/news/34845](https://hyper.ai/cn/news/34845)

* **科研团队：** 上海交通大学 X-LANCE 实验室吴梦玥老师团队、德克萨斯大学阿灵顿分校 UTA 、天桥脑科学研究院 (TCCI) 和 ThetaAI 公司

* **相关研究：**  搭建了一个新型的对话 Agent 模拟系统、D4 数据集、三层记忆存储结构和全新的记忆检索机制、患者 Agent、精神科医生 Agent、指导员 Agent，提升抑郁症与自杀倾向诊断准确率

* **发布期刊：** arXiv, 2024.9

* **论文链接：** [Depression Diagnosis Dialogue Simulation: Self-improving Psychiatrist with Tertiary Memory](https://arxiv.org/abs/2409.15084)

### **29. [深度学习模型 LucaProt，助力 RNA 病毒识别](https://hyper.ai/cn/news/34968)**
* **中文解读：** [https://hyper.ai/cn/news/34968](https://hyper.ai/cn/news/34968)

* **科研团队：** 中山大学医学院的施莽教授、浙江大学、复旦大学、中国农业大学、香港城市大学、广州大学、悉尼大学、阿里云飞天实验室

* **相关研究：**  云计算与 AI 技术、宏基因组挖掘技术、NCBI SRA 数据库、CNGBdb 数据库、基于数据驱动的深度学习模型 LucaProt、Transformer 框架、大模型表征技术、揭露了 161,979 种潜在 RNA 病毒物种和 180 个病毒超群的存在

* **发布期刊：** Cell, 2024.9

* **论文链接：** [Using artificial intelligence to document the hidden RNA virosphere](https://doi.org/10.1016/j.cell.2024.09.027)
### **30. [医学图像预训练框架 UniMedI，打破医学数据异构化藩篱](https://hyper.ai/cn/news/35128)**
* **中文解读：** [https://hyper.ai/cn/news/35128](https://hyper.ai/cn/news/35128)

* **科研团队：** 浙江大学胡浩基团队、微软亚洲研究院邱锂力团队

* **相关研究：**  「伪配对」(Pseudo-Pairs) 技术、MIMIC-CXR 2.0.0 数据集、BIMCV 数据集、预训练 UniMedI 框架、ViT-B/16 视觉编码器 、BioClinicalBERT 文本编码器 、VL (Vision-Language) 对比学习、辅助任务设计、UniMiss 医学自我监督表达学习框架

* **发布期刊：** ECCV 2024

* **论文链接：** [Unified Medical Image Pre-training in Language-Guided Common Semantic Space](https://eccv.ecva.net/virtual/2024/poster/1165)
### **31. [多语言医学大模型 MMed-Llama 3，更加适配医疗应用场景](https://hyper.ai/cn/news/35242)**
* **中文解读：** [https://hyper.ai/cn/news/35242](https://hyper.ai/cn/news/35242)

* **科研团队：** 上海交通大学王延峰教授与谢伟迪教授团队

* **相关研究：**  多语言医疗语料库 MMedC、多语言医疗问答评测标准 MMedBench、基座模型 MMed-Llama 3、MMedLM 多语言模型、MMedLM 2 多语言模型、 MMed-Llama 3 多语言模型

* **发布期刊：** Nature Communications, 2024.9

* **论文链接：** [Towards building multilingual language model for medicine](https://www.nature.com/articles/s41467-024-52417-z)
### **32. [胶囊内窥镜图像拼接方法 S2P-Matching，助力胶囊内窥镜图像拼接](https://hyper.ai/cn/news/35313)**
* **中文解读：** [https://hyper.ai/cn/news/35313](https://hyper.ai/cn/news/35313)

* **科研团队：** 华中科技大学陆枫团队、上海交通大学盛斌、中南民族大学、香港科技大学（广州）分校、香港理工大学、悉尼大学、匹配正确率提升 187.9%

* **相关研究：**  胶囊内窥镜图像拼接方法 S2P-Matching、自监督对比学习方法、双分支编码器提取局部特征、Transformer 模型、结合数据增强、对比学习、像素级匹配

* **发布期刊：** IEEE Transactions on Biomedical Engineering, 2024.9

* **论文链接：** [S2P-Matching: Self-supervised Patch-based Matching Using Transformer for Capsule Endoscopic Images Stitching](http://dx.doi.org/10.1109/TBME.2024.3462502)

## **AI+ 材料化学：AI+Materials Chemistry**

### **1. [高通量计算框架 33 分钟生成 12 万种新型 MOFs 候选材料](https://hyper.ai/news/30269)**
* **中文解读：** [https://hyper.ai/news/30269](https://hyper.ai/news/30269)

* **科研团队：** 美国阿贡国家实验室 Eliu A. Huerta 研究团队

* **相关研究：** hMOFs 数据集、生成式 AI、GHP-MOFsassemble、MMPA、DiffLinker、CGCNN、GCMC

* **发布期刊：** Nature, 2024.02

* **论文链接：** [A generative artificial intelligence framework based on a molecular diffusion model for the design of metal-organic frameworks for carbon capture](https://www.nature.com/articles/s42004-023-01090-2)

### **2. [机器学习算法模型筛选 P-SOC 电极材料](https://hyper.ai/news/29069)**
* **中文解读：** [https://hyper.ai/news/29069](https://hyper.ai/news/29069)

* **科研团队：** 广州大学叶思宇研究团队

* **相关研究：** XGBoost、机器学习模型、RF、DFT。成功筛选电极材料 LCN91

* **发布期刊：** ADVANCED FUNCTIONAL MATERIALS, 2023.12

* **论文链接：** [Machine-Learning Assisted Screening Proton Conducting Co/Fe based Oxide for the Air Electrode of Protonic Solid Oxide Cell](https://onlinelibrary.wiley.com/doi/10.1002/adfm.202309855)

### **3. [SEN 机器学习模型，实现高精度的材料性能预测](https://hyper.ai/news/28410)**
* **中文解读：** [https://hyper.ai/news/28410](https://hyper.ai/news/28410)

* **科研团队：** 中山大学李华山、王彪课题组

* **相关研究：** Materials Project 数据库、SEN、capsule mechanism、深度学习。SEN 模型预测带隙和形成能的平均绝对误差，分别比常见机器学习模型低约 22.9% 和 38.3%。

* **发布期刊：** Nature Communications, 2023.08

* **论文链接：** [Material symmetry recognition and property prediction accomplished by crystal capsule representation](https://www.nature.com/articles/s41467-023-40756-2)

### **4. [深度学习工具 GNoME 发现 220 万种新晶体](https://hyper.ai/news/28347)**
* **中文解读：** [https://hyper.ai/news/28347](https://hyper.ai/news/28347)

* **科研团队：** 谷歌 DeepMind 研究团队

* **相关研究：** GNoME 数据库、GNoME、SOTA GNN 模型、深度学习、Materials Project、OQMD、WBM、ICSD

* **发布期刊：** Nature, 2023.11

* **论文链接：** [Scaling deep learning for materials discovery](https://www.nature.com/articles/s41586-023-06735-9)

### **5. [场诱导递归嵌入原子神经网络可准确描述外场强度、方向变化](https://hyper.ai/news/28285)**
* **中文解读：** [https://hyper.ai/news/28285](https://hyper.ai/news/28285)

* **科研团队：** 中国科学技术大学的蒋彬课题组

* **相关研究：** 场诱导递归嵌入原子神经网络 FIREANN、FIREANN-wF 模型。可准确描述外场强度和方向的变化时系统能量的变化趋势，还能对任意阶数的系统响应进行预测

* **发布期刊：** Nature Communication, 2023.10

* **论文链接：** [Universal machine learning for the response of atomistic systems to external fields](https://www.nature.com/articles/s41467-023-42148-y)

### **6. [机器学习预测多孔材料水吸附等温线](https://hyper.ai/news/28260)**
* **中文解读：** [https://hyper.ai/news/28260](https://hyper.ai/news/28260)

* **科研团队：** 华中科技大学的李松课题组

* **相关研究：** EWAID 数据库、机器学习模型、RF、ANN。RF 预测水吸附等温线有高精度和高灵敏度

* **发布期刊：** Journal of Materials Chemistry A, 2023.09

* **论文链接：** [Machine learning-assisted prediction of water adsorption isotherms and cooling performance](https://pubs.rsc.org/en/content/articlelanding/2023/TA/D3TA03586G)

### **7. [利用机器学习优化 BiVO(4) 光阳极的助催化剂](https://hyper.ai/news/28013)**
* **中文解读：** [https://hyper.ai/news/28013](https://hyper.ai/news/28013)

* **科研团队：** 清华大学朱宏伟课题组

* **相关研究：** ML、神经网络、AdaBoost 算法、Gradient Boosting、自解释模型、Bagging 算法、交叉验证

* **发布期刊：** Journal of Materials Chemistry A, 2023.10

* **论文链接：** [A comprehensive machine learning strategy for designing high-performance photoanode catalysts](https://pubs.rsc.org/en/content/articlelanding/2023/TA/D3TA04148D)

### **8. [RetroExplainer 算法基于深度学习进行逆合成预测](https://hyper.ai/news/27406)**
* **中文解读：** [https://hyper.ai/news/27406](https://hyper.ai/news/27406)

* **科研团队：** 山东大学、电子科技大学课题组

* **相关研究：** RetroExplainer、深度学习、MSMS-GT、DAMT、可解释的决策模块、路线预测模块。RetroExplainer 提出的合成路线中，86.9% 的反应得到了文献的验证

* **发布期刊：** Nature Communications, 2023.10

* **论文链接：** [Retrosynthesis prediction with an interpretable deep-learning framework based on molecular assembly tasks](https://www.nature.com/articles/s41467-023-41698-5)

### **9. [深度神经网络+自然语言处理，开发抗蚀合金](https://hyper.ai/news/25891)**
* **中文解读：** [https://hyper.ai/news/25891](https://hyper.ai/news/25891)

* **科研团队：** 德国马克思普朗克铁研究所的研究团队

* **相关研究：** DNN、NLP。读取有关合金加工和测试方法的文本数据，有预测新元素的能力

* **发布期刊：** Science Advances, 2023.08

* **论文链接：** [Enhancing corrosion-resistant alloy design through natural language processing and deep learning](https://www.science.org/doi/10.1126/sciadv.adg7992)

### **10. [深度学习通过表面观察确定材料的内部结构](https://hyper.ai/news/25859)**
* **中文解读：** [https://hyper.ai/news/25859](https://hyper.ai/news/25859)

* **科研团队：** 麻省理工学院的研究团队

* **相关研究：** 深度学习、FEA 计算、Abaqus 可视化工具、GAN、ViViT、CNN

* **发布期刊：** Advanced Materials, 2023.03

* **论文链接：** [Fill in the Blank: Transferrable Deep Learning Approaches to Recover Missing Physical Field Information](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202301449)

### **11. [利用创新 X 射线闪烁体开发 3 种新材料](https://hyper.ai/news/31465)**
* **中文解读：** [https://hyper.ai/news/31465](https://hyper.ai/news/31465)

* **科研团队：** 河北大学张海磊研究团队

* **相关研究：** 水分散性 X 射线闪烁体、纳米材料、聚氨酯泡沫、X 射线成像柔性水凝胶闪烁体屏幕、多级防伪信息加密的复合水凝胶

* **发布期刊：** Nature Communications, 2024.03

* **论文链接：** [Water-dispersible X-ray scintillators enabling coating and blending with polymer materials for multiple applications](https://www.nature.com/articles/s41467-024-46287-8)

### **12. [半监督学习提取无标签数据中的隐藏信息](https://hyper.ai/news/31089)**
* **中文解读：** [https://hyper.ai/news/31089](https://hyper.ai/news/31089)

* **科研团队：** 上海交大万佳雨研究团队研究团队

* **相关研究：** 半监督学习、无标签数据、贝叶斯协同训练、部分视图模型、完整视图模型、锂电池寿命预测精度提升 20%

* **发布期刊：** Joule, 2024.03

* **论文链接：** [Semi-supervised learning for explainable few-shot battery lifetime prediction](https://doi.org/10.1016/j.joule.2024.02.020 )

### **13. [基于自动机器学习进行知识自动提取](https://hyper.ai/news/30920)**
* **中文解读：** [https://hyper.ai/news/30920](https://hyper.ai/news/30920)

* **科研团队：** 上海交大贺玉莲研究团队

* **相关研究：** 自动机器学习AutoML、催化剂、化学吸附能、Eads  值、特征删除实验、神经网络、高通量密度泛函理论

* **发布期刊：** PNAS, 2024.03

* **论文链接：** [Interpreting chemisorption strength with AutoML-based feature deletion experiments](https://hyper.ai/news/30920)

### **14. [一种三维 MOF 材料吸附行为预测的机器学习模型 Uni-MOF](https://hyper.ai/news/30663)**
* **中文解读：** [https://hyper.ai/news/30663](https://hyper.ai/news/30663)

* **科研团队：** 清华大学化工系卢滇楠研究团队

* **相关研究：** hMOFs50 数据库、MOF/COF 数据库、微调 Uni-MOF、在识别超过 63 万个三维空间构型及其原子间连接关系上的有效性

* **发布期刊：** Nature Communications, 2024.03

* **论文链接：** [A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal-organic frameworks](https://www.nature.com/articles/s41467-024-46276-x)

### **15. [微电子加速迈向后摩尔时代！集成 DNN 与纳米薄膜技术，精准分析入射光角度](https://hyper.ai/news/32326)**
* **中文解读：** [https://hyper.ai/news/32326](https://hyper.ai/news/32326)

* **科研团队：** 复旦大学梅永丰课题组

* **相关研究：** 有限元模型、应变纳米膜释放模型、菲克定律、深度神经网络、三维光探测器、角度敏感检测模型

* **发布期刊：** Nature Communications, 2024.04

* **论文链接：** [Multilevel design and construction in nanomembrane rolling for three-dimensional angle-sensitive photodetection](https://www.nature.com/articles/s41467-024-47405-2)

### **16. [重塑锂电池性能边界，基于集成学习提出简化电化学模型](https://hyper.ai/news/32323)**
* **中文解读：** [https://hyper.ai/news/32323](https://hyper.ai/news/32323)

* **科研团队：** 武汉理工大学康健强团队

* **相关研究：** 简化电化学模型、集成学习模型、机器学习、一阶惯性元件 FIE、离散时间实现算法 DRA、分数阶帕德逼近 FOM、三参数抛物线近似 TPM

* **发布期刊：** iScience, 2024.05

* **论文链接：** [A simplified electrochemical model forlithium-ion batteries based on ensemblelearning](https://www.sciencedirect.com/science/article/pii/S2589004224009076)

### **17. [最强铁基超导磁体诞生！基于机器学习，磁场强度超过先前记录 2.7 倍](https://hyper.ai/news/32556)**
* **中文解读：** [https://hyper.ai/news/32556](https://hyper.ai/news/32556)

* **科研团队：** 东京农工大学研究团队

* **相关研究：** BOXVIA 机器学习、数据驱动循环、数值模拟、铁基超导永磁体 Ba122、场冷磁化 (FCM) 模型

* **发布期刊：** NPG Asia Materials, 2024.06

* **论文链接：** [Superstrength permanent magnets with iron-based superconductors by data- and researcher-driven process design](https://www.nature.com/articles/s41427-024-00549-5)

### **18. [神经网络替代密度泛函理论！通用材料模型实现超精准预测](https://hyper.ai/news/32891)**
* **中文解读：** [https://hyper.ai/news/32891](https://hyper.ai/news/32891)

* **科研团队：** 清华大学物理系的徐勇、段文晖团队

* **相关研究：** Materials Project 数据库、深度学习密度泛函理论哈密顿量 (DeepH) 方法、通用材料模型、神经网络、等变神经网络、自动化交互式基础设施和数据库 (AiiDA) 框架

* **发布期刊：** Science Bulletin, 2024.06

* **论文链接：** [Universal materials model of deep-learning density functional theory Hamiltonian](https://doi.org/10.1016/j.scib.2024.06.011)

### **19. [神经网络密度泛函框架打开物质电子结构预测的黑箱](https://hyper.ai/news/33525)**
* **中文解读：** [https://hyper.ai/news/33525](https://hyper.ai/news/33525)

* **科研团队：** 清华大学徐勇、段文晖课题组

* **相关研究：** 神经网络密度泛函理论、变分密度泛函理论、等价神经网络、Julia 语言、Zygote 自动微分框架、深度学习、无监督学习、DFT

* **发布期刊：** Phys. Rev. Lett., 2024.08

* **论文链接：** [Neural-network density functional theory based on variational energy minimization](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.076401)

### **20. [用神经网络首创全前向智能光计算训练架构，国产光芯片实现重大突破](https://hyper.ai/news/33440)**
* **中文解读：** [https://hyper.ai/news/33440](https://hyper.ai/news/33440)

* **科研团队：** 清华大学戴琼海院士、方璐教授研究团队

* **相关研究：** 神经网络、全前向模式、机器学习、MNIST 数据集、Fashion-MNIST 数据集、CIFAR-10 数据集、ImageNet 数据集、MWD 数据集、鸢尾花数据集、Chromium target 数据集

* **发布期刊：** Nature, 2024.08

* **论文链接：** [Fully forward mode training for optical neural networks](https://www.nature.com/articles/s41586-024-07687-4)
### **21. [化学大语言模型 ChemLLM 覆盖 7 百万问答数据，专业能力比肩 GPT-4](https://hyper.ai/news/34170)**
* **中文解读：** [https://hyper.ai/news/34170](https://hyper.ai/news/34170)

* **科研团队：** 上海人工智能实验室

* **相关研究：**  大规模化学数据集 ChemData 、ChemPref-10K 的中英文版本数据集、C- MHChem 数据集、ChemBench4K 化学能力评测基准数据集、大规模化学基准测试 ChemBench、Multi-Corpus 综合语料库、NLP 任务、化学大语言模型

* **发布期刊：** arXiv, 2024.02

* **论文链接：** [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852)
### **22. [可晶圆级生产的人工智能自适应微型光谱仪](https://hyper.ai/news/34075)**
* **中文解读：** [https://hyper.ai/news/34075](https://hyper.ai/news/34075)

* **科研团队：** 复旦大学材料科学系、智慧纳米机器人与纳米系统国际研究院梅永丰教授课题组

* **相关研究：**  光学光谱仪、微型化重构光谱仪、CMOS 集成电路工艺、窄带通道电流数据集 、全部通道电流数据集、在整个可见光波段表现出准确的光谱重构能力

* **发布期刊：** PNAS,  2024.08

* **论文链接：** [CMOS-Compatible Reconstructive Spectrometers with Self-Referencing Integrated Fabry-Perot Resonatorsl](https://www.pnas.org/doi/10.1073/pnas.2403950121)
### **23. [GNNOpt 模型，识别数百种太阳能电池和量子候选材料](https://hyper.ai/cn/news/35009)**
* **中文解读：** [https://hyper.ai/cn/news/35009](https://hyper.ai/cn/news/35009)

* **科研团队：** 日本东北大学、麻省理工学院
* **相关研究：**  DFT 计算、人工智能工具 GNNOpt、「集成嵌入」技术、集成等变神经网络、Materials Project 数据库、自动嵌入优化的集成嵌入层、成功识别出 246 种太阳能转换效率超过 32% 的材料、以及 296 种具有高量子权重的量子材料

* **发布期刊：** Advanced Materials, 2024.6

* **论文链接：** [Universal Ensemble-Embedding Graph Neural Network for Direct Prediction of Optical Spectra from Crystal Structures](https://onlinelibrary.wiley.com/doi/epdf/10.1002/adma.202409175)

## **AI+动植物科学：AI+Zoology-Botany**

### **1. [SBeA 基于少样本学习框架进行动物社会行为分析](https://hyper.ai/news/29353)**
* **中文解读：** [https://hyper.ai/news/29353](https://hyper.ai/news/29353)

* **科研团队：** 中科院深圳先进院蔚鹏飞研究团队

* **相关研究：** PAIR-R24M 数据集、双向迁移学习、非监督式学习、人工神经网络、身份识别模型。在多动物身份识别方面的准确率超过 90% 

* **发布期刊：** Nature Machine Intelligence, 2024.01

* **论文链接：** [Multi-animal 3D social pose estimation, identification and behaviour embedding with a few-shot learning framework](https://www.nature.com/articles/s42256-023-00776-5)

### **2. [基于孪生网络的深度学习方法，自动捕捉胚胎发育过程](https://hyper.ai/news/28787)**
* **中文解读：** [https://hyper.ai/news/28787](https://hyper.ai/news/28787)

* **科研团队：** 系统生物学家 Patrick Müller 及康斯坦茨大学研究团队

* **相关研究：** ImageNet 数据集、孪生网络、深度学习、迁移学习、三联体损失训练、迭代训练、分任务训练。在没有人为干预的情况下识别胚胎发育特征阶段点

* **发布期刊：** Nature Methods, 2023.11

* **论文链接：** [Uncovering developmental time and tempo using deep learning](https://www.nature.com/articles/s41592-023-02083-8)

### **3. [利用无人机采集植物表型数据的系统化流程，预测最佳采收日期](https://hyper.ai/news/28303)**
* **中文解读：** [https://hyper.ai/news/28303](https://hyper.ai/news/28303)

* **科研团队：** 东京大学和千叶大学的研究团队

* **相关研究：** 利润预测模型、分割模型、交互式标注、LabelMe、非线性回归模型、BiSeNet 模型
* **发布期刊：** Plant Phenomics, 2023.09

* **论文链接：** [Drone-Based Harvest Data Prediction Can Reduce On-Farm Food Loss and Improve Farmer Income](https://spj.science.org/doi/10.34133/plantphenomics.0086#body-ref-B4)

### **4. [AI 相机警报系统准确区分老虎和其他物种](https://hyper.ai/news/27954)**
* **中文解读：** [https://hyper.ai/news/27954](https://hyper.ai/news/27954)

* **科研团队：** 克莱姆森大学的研究团队

* **相关研究：** TrailGuard AI。1 分钟内将相关图像传到保护区管理员的终端设备上

* **发布期刊：** BioScience, 2023.09

* **论文链接：** [Accurate proteome-wide missense variant effect prediction with AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492)

### **5. [利用拉布拉多猎犬数据，对比 3 种模型，发现了影响嗅觉检测犬表现的行为特性](https://hyper.ai/news/25472)**
* **中文解读：** [https://hyper.ai/news/25472](https://hyper.ai/news/25472)

* **科研团队：** 美国全国儿童医院阿比盖尔·韦克斯纳研究所、洛基维斯塔大学的研究团队

* **相关研究：** AT 测试、Env 测试、随机森林、支持向量机、逻辑回归、PCA、RFECV

* **发布期刊：** Scientific Reports, 2023.08

* **论文链接：** [Machine learning prediction and classification of behavioral selection in a canine olfactory detection program](https://www.nature.com/articles/s41598-023-39112-7)

### **6. [基于人脸识别 ArcFace Classification Head 的多物种图像识别模型](https://hyper.ai/news/25164)**
* **中文解读：** [https://hyper.ai/news/25164](https://hyper.ai/news/25164)

* **科研团队：** 夏威夷大学的研究团队

* **相关研究：** [鲸类数据集](https://github.com/knshnb/kaggle-happywhale-1st-place)、图像修剪模型、图像识别模型、YOLOv5、Detic。达到了 0.869 的平均准确率

* **发布期刊：** Methods in Ecology and Evolution, 2023.07

* **论文链接：** [A deep learning approach to photo–identification demonstrates high performance on two dozen cetacean species](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14167)

### **7. [利用 Python API 与计算机视觉 API，监测日本的樱花开放情况](https://hyper.ai/news/24512)**
* **中文解读：** [https://hyper.ai/news/24512](https://hyper.ai/news/24512)

* **科研团队：** 澳大利亚莫纳什大学的研究团队

* **相关研究：** 社交网站 (SNS) 数据、Google Cloud Vision AI、机器学习模型
* **发布期刊：** Flora, 2023.07

* **论文链接：** [The spatiotemporal signature of cherry blossom flowering across Japan revealed via analysis of social network site images](https://www.sciencedirect.com/science/article/abs/pii/S0367253023001019)

### **8. [基于机器学习的群体遗传方法，揭示葡萄风味的形成机制](https://hyper.ai/news/24442)**
* **中文解读：** [https://hyper.ai/news/24442](https://hyper.ai/news/24442)

* **科研团队：** 中国农业科学院深圳农业基因组的研究团队

* **相关研究：** [葡萄基因组序列](https://github.com/zhouyflab/Grapevine_Adaptive_Maladaptive_Introgression)、机器学习

* **发布期刊：** Proceedings of the National Academy of Sciences, 2023.06

* **论文链接：** [Adaptive and maladaptive introgression in grapevine domestication](https://www.pnas.org/doi/abs/10.1073/pnas.2222041120)


### **9. [综述：借助 AI 更高效地开启生物信息学研究](https://hyper.ai/news/33931)**
* **中文解读：** [https://hyper.ai/news/33931](https://hyper.ai/news/33931)

* **主要内容：** AI 在同源搜索、多重比对及系统发育构建、基因组序列分析、基因发现等生物学领域中，都有丰富的应用案例。作为一名生物学研究人员，能熟练地将机器学习工具整合到数据分析中，必将加速科学发现、提升科研效率。

### **10. [BirdFlow 模型准确预测候鸟的飞行路径](https://hyper.ai/cn/news/34781)**
* **中文解读：** [https://hyper.ai/news/33942](https://hyper.ai/news/33942)

* **科研团队：** 马萨诸塞州立大学、康奈尔大学的研究团队

* **相关研究：** 计算机建模、eBird 数据集、马尔可夫模型、Hyperparameter grid search、Entropy calibration、k-week  forecasting

* **发布期刊：** Methods in Ecology and Evolution, 2023.01

* **论文链接：** [BirdFlow: Learning seasonal bird movements from eBird data](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14052)
### **11. [新的鲸鱼生物声学模型，可识别 8 种鲸类](https://hyper.ai/cn/news/34781)**
* **中文解读：** [https://hyper.ai/cn/news/34781](https://hyper.ai/cn/news/34781)

* **科研团队：** Google Research 团队

* **相关研究：** 梅尔尺度的频率轴 、压缩数振幅、可通过 TensorFlow 的 SavedModel API 独立调用、卷积神经网络、用于检测座头鲸叫声的分类模型、互动可视化工具「Pattern Radio」、专门用于识别蓝鲸和长须鲸的模型、可识别目前已知 94 种鲸鱼种类中的 8 个不同物种

* **发布期刊：** Google Research, 2024.9

* **论文链接：** [Whistles, songs, boings, and biotwangs: Recognizing whale vocalizations with AI](https://research.google/blog/whistles-songs-boings-and-biotwangs-recognizing-whale-vocalizations-with-ai)

## **AI+农林畜牧业：AI+Agriculture-Forestry-Animal husbandry**

### **1. [利用卷积神经网络，对水稻产量进行迅速、准确的统计](https://hyper.ai/news/26100)**
* **中文解读：** [https://hyper.ai/news/26100](https://hyper.ai/news/26100)

* **科研团队：** 京都大学的研究团队

* **相关研究：** 卷积神经网络。CNN 模型可以对不同拍摄角度、时间和时期下得到的农田照片准确分析，得到稳定的产量预测结果

* **发布期刊：** Plant Phenomics, 2023.07

* **论文链接：** [Deep Learning Enables Instant and Versatile Estimation of Rice Yield Using Ground-Based RGB Images](https://spj.science.org/doi/10.34133/plantphenomics.0073)

### **2. [通过 YOLOv5 算法，设计监测母猪姿势与猪仔出生的模型](https://hyper.ai/news/25131)**
* **中文解读：** [https://hyper.ai/news/25131](https://hyper.ai/news/25131)

* **科研团队：** 南京农业大学研究团队

* **相关研究：** YOLOv5、检测母猪姿势和仔猪的模型。能够在产仔开始前 5 小时发出警报，总体平均准确率为 92.9%

* **发布期刊：** Sensors, 2023.01

* **论文链接：** [Sow Farrowing Early Warning and Supervision for Embedded Board Implementations](https://www.mdpi.com/1424-8220/23/2/727)

### **3. [结合实验室观测与机器学习，证明番茄与烟草植物在胁迫环境下发出的超声波能在空气中传播](https://hyper.ai/news/24547)**
* **中文解读：** [https://hyper.ai/news/24547](https://hyper.ai/news/24547)

* **科研团队：** 以色列特拉维夫大学的研究团队

* **相关研究：** 机器学习模型、SVM、Basic、MFCC、Scattering network、神经网络模型、留一法交叉验证。识别准确率高达 99.7%、4-6 天时番茄尖叫声最大

* **发布期刊：** Cell，2023.03

* **论文链接：** [Sounds emitted by plants under stress are airborne and informative](https://doi.org/10.1016/j.cell.2023.03.009)

### **4. [无人机+ AI 图像分析，检测林业害虫](https://hyper.ai/news/23807)**
* **中文解读：** [https://hyper.ai/news/23807](https://hyper.ai/news/23807)

* **科研团队：** 里斯本大学研究团队

* **相关研究：** FRCNN、YOLO 模型。YOLO 模型的检测性能高于 FRCNN、无人机和 AI 模型相结合能够有效地对松异舟蛾巢穴进行早期检测

* **发布期刊：** NeoBiota, 2023.05

* **论文链接：** [Testing early detection of pine processionary moth Thaumetopoea pityocampa nests using UAV-based methods](https://neobiota.pensoft.net/article/95692/)

### **5. [计算机视觉+深度学习开发奶牛跛行检测系统](https://hyper.ai/news/33957)**
* **中文解读：** [https://hyper.ai/news/33957](https://hyper.ai/news/33957)

* **科研团队：** 纽卡斯尔大学及费拉科学有限公司的研究团队

* **相关研究：** 计算机视觉、深度学习、Mask-RCNN 算法、SORT 算法、CatBoost 算法。准确度可达 94%-100%

* **发布期刊：** Nature, 2023.03

* **论文链接：** [Deep learning pose estimation for multi-cattle lameness detection](https://www.nature.com/articles/s41598-023-31297-1)

## **AI+ 气象学：AI+Meteorology**
### **1. [综述：数据驱动的机器学习天气预报模型](https://hyper.ai/news/28124)**
* **中文解读：** [https://hyper.ai/news/28124](https://hyper.ai/news/28124)

* **主要内容：** 数值天气预报是天气预报的主流方法。它通过数值积分，对地球系统的状态进行逐网格的求解，是一个演绎推理的过程。 2022 年以来，天气预报领域的机器学习模型取得了一系列突破，部分成果可以与欧洲中期天气预报中心的高精度预测匹敌。

### **2. [综述：从雹暴中心收集数据，利用大模型预测极端天气](https://hyper.ai/news/25874)**
* **中文解读：** [https://hyper.ai/news/25874](https://hyper.ai/news/25874)

* **主要内容：** 2021 年，达摩院与国家气象中心联合研发了 AI 算法用于天气预测，并成功预测了多次强对流天气。同年 9 月，Deepmind 在《Nature》上发表文章，利用深度生成模型进行降雨量的实时预报。

 2023 年年初，Deepmind 正式推出了 GraphCast，可以在一分钟内对全球未来 10 天的气象，进行分辨率为 0.25° 的预测。 4 月，南京信息工程大学和上海人工智能实验室合作研发了「风乌」气象预测大模型，误差较 GraphCast 进一步降低。

 随后，华为推出了「盘古」气象大模型。由于模型中引出了三维神经网络，「盘古」的预测准确率首次超过了目前最准确的 NWP 预测系统。近期，清华大学和复旦大学相继发布了「NowCastNet」和「伏羲」模型。

### **3. [利用全球风暴解析模拟与机器学习，创建新算法，准确预测极端降水](https://hyper.ai/news/24995)**
* **中文解读：** [https://hyper.ai/news/24995](https://hyper.ai/news/24995)

* **科研团队：** 哥伦比亚大学 LEAP 实验室

* **相关研究：** 机器学习、Baseline-NN、Org-NN、神经网络
* **发布期刊：** PNAS, 2023.03

* **论文链接：** [Implicit learning of convective organization explains precipitation stochasticity
](https://www.pnas.org/doi/10.1073/pnas.2216158120)


### **4. [基于随机森林的机器学习模型 CSU-MLP，预测中期恶劣天气](https://hyper.ai/news/33966)**
* **中文解读：** [https://hyper.ai/news/33966](https://hyper.ai/news/33966)

* **科研团队：** 美国科罗拉多州立大学和国家海洋和大气管理局的研究团队

* **相关研究：** GEFS/R 数据集、机器学习、插值处理、RF。可对中期（4-8 天）范围内恶劣天气进行准确预报

* **发布期刊：** Weather and Forecasting, 2022.08

* **论文链接：** [A new paradigm for medium-range severe weather forecasts: probabilistic random forest-based predictions](https://arxiv.org/abs/2208.02383)

## **AI+ 天文学：AI+Astronomy**
### **1. [PRIMO 算法学习黑洞周围的光线传播规律，重建出更清晰的黑洞图像](https://hyper.ai/news/23698)**
* **中文解读：** [https://hyper.ai/news/23698](https://hyper.ai/news/23698)

* **科研团队：** 普林斯顿高等研究院研究团队

* **相关研究：** PRIMO 算法、PCA、GRMHD。PRIMO 重建黑洞图像

* **发布期刊：** The Astrophysical Journal Letters, 2023.04

* **论文链接：** [The Image of the M87 Black Hole Reconstructed with PRIMO](https://iopscience.iop.org/article/10.3847/2041-8213/acc32d/pdf)


### **2. [利用模拟数据训练计算机视觉算法，对天文图像进行锐化「还原」](https://hyper.ai/news/33975)**
* **中文解读：** [https://hyper.ai/news/33975](https://hyper.ai/news/33975)

* **科研团队：** 清华大学及美国西北大学研究团队

* **相关研究：** [Galsim](https://github.com/GalSim-developers/GalSim)、[COSMOS](https://doi.org/10.5281/zenodo.3242143)、计算机视觉算法、CNN、Richardson-Lucy 算法、unrolled-ADMM 神经网络

* **发布期刊：** 皇家天文学会月刊，2023.06

* **论文链接：** [Galaxy image deconvolution for weak gravitational lensing with unrolled plug-and-play ADMM](https://www.nature.com/articles/s41421-023-00543-1)

### **3. [利用无监督机器学习算法 Astronomaly ，找到了之前为人忽视的异常现象](https://hyper.ai/news/26316)**
* **中文解读：** [https://hyper.ai/news/26316](https://hyper.ai/news/26316)

* **科研团队：** 西开普大学的研究者

* **相关研究：** CNN、无监督机器学习、Astronomaly、PCA、孤立森林、LOF 算法、iForest 算法、NS 算法、DR 算法。Astronomaly 从异常评分最高的 2,000 张图像中找到了 1,635 处异常

* **发布期刊：** arXiv, 2023.09

* **论文链接：** [Astronomaly at Scale: Searching for Anomalies Amongst 4 Million Galaxies](https://arxiv.org/abs/2309.08660)

### **4. [基于机器学习的 CME 识别与参数获取方法](https://hyper.ai/news/31870)**
* **中文解读：** [https://hyper.ai/news/31870](https://hyper.ai/news/31870)

* **科研团队：** 中国科学院国家空间科学中心太阳活动与空间天气重点实验室的研究团队

* **相关研究：** 机器学习、神经网络、Otsu 算法、轨迹匹配算法、自动识别、参数获取、CACTus 、 CORIMP 、 SEEDS。可识别日冕物质抛射

* **发布期刊：** THE ASTROPHYSICAL JOURNAL, 2024.04

* **论文链接：** [An Algorithm for the Determination of Coronal Mass Ejection Kinematic Parameters Based on Machine Learning](https://iopscience.iop.org/article/10.3847/1538-4365/ad2dea)

### **5. [深度学习发现 107 例中性碳吸收线](https://hyper.ai/news/32210)**
* **中文解读：** [https://hyper.ai/news/32210](https://hyper.ai/news/32210)

* **科研团队：** 中国科学院上海天文台研究员葛健带领的国际团队

* **相关研究：** 深度学习方法、SDSS DR12、卷积神经网络模型、发现了 107 例宇宙早期中性碳吸收线，探测精度达 99.8%

* **发布期刊：** MNRAS, 2024.05

* **论文链接：** [Detecting rare neutral atomic-carbon absorbers with a deep neuralnetwork](https://doi.org/10.1093/mnras/stae799)
### **6. [StarFusion 模型实现高空间分辨率图像的预测](https://hyper.ai/news/34254)**
* **中文解读：** [https://hyper.ai/news/34254](https://hyper.ai/news/34254)

* **科研团队：** 北京师范大学地表过程与资源生态国家重点实验室陈晋团队

* **相关研究：** 深度学习方法、遥感影像、高空间分辨率图像的预测、提出了双流时空解耦融合架构模型 StarFusion、Gaofen-1 数据集、Sentinel-2 卫星数据集、SRGAN-STF 模型、线性回归模型、多变量回归关系模型

* **发布期刊：** Journal of Remote Sensing, 2024.07

* **论文链接：** [A Hybrid Spatiotemporal Fusion Method for High Spatial Resolution Imagery: Fusion of Gaofen-1 and Sentinel-2 over Agricultural Landscapes](https://spj.science.org/doi/10.34133/remotesensing.0159)

## **AI+ 自然灾害：AI+Natural Disaster**
### **1. [机器学习预测未来 40 年的地面沉降风险](https://hyper.ai/news/30173)**
* **中文解读：** [https://hyper.ai/news/30173](https://hyper.ai/news/30173)

* **科研团队：** 中南大学柳建新研究团队

* **相关研究：** SAR 数据集、机器学习模型、XGBR、LSTM

* **发布期刊：** Journal of Environmental Management, 2024.02

* **论文链接：** [Machine learning-based techniques for land subsidence simulation in an urban area](https://www.sciencedirect.com/science/article/abs/pii/S0301479724000641?via%3Dihub
)

### **2. [语义分割模型 SCDUNet++ 用于滑坡测绘](https://hyper.ai/news/29672)**
* **中文解读：** [https://hyper.ai/news/29672](https://hyper.ai/news/29672)

* **科研团队：** 成都理工大学刘瑞研究团队

*  **相关研究：** Sentinel-2 多光谱数据、NASADEM 数据、滑坡数据、GLFE、CNN、DSSA、DSC、DTL、Transformer、深度迁移学习。交并比提高了 1.91% - 24.42%，F1 提高了 1.26% - 18.54%

* **发布期刊：** International Journal of Applied Earth Observation and Geoinformation, 2024.01

* **论文链接：** [A deep learning system for predicting time to progression of diabetic retinopathy](https://www.nature.com/articles/s41591-023-02702-z)

### **3. [神经网络将太阳二维图像转为三维重建图像](https://hyper.ai/news/28797)**
* **中文解读：** [https://hyper.ai/news/28797](https://hyper.ai/news/28797)

* **科研团队：** 科罗拉多州国家大气研究中心

*  **相关研究：** NeRFs 神经网络、SuNeRF 模型。首次揭示了太阳的两极

* **发布期刊：** arxiv, 2022.11

* **论文链接：** [SuNeRF: Validation of a 3D Global Reconstruction of the Solar Corona Using Simulated EUV Images](https://arxiv.org/abs/2211.14879)

### **4. [可叠加神经网络分析自然灾害中的影响因素](https://hyper.ai/news/24957)**
* **中文解读：** [https://hyper.ai/news/24957](https://hyper.ai/news/24957)

* **科研团队：** 加利福尼亚大学洛杉矶分校的研究团队

* **相关研究：** 可叠加神经网络、半自动检测算法、additive ANN、SNN、特征选择模型、多阶段训练
* **发布期刊：** Communications Earth & Environment, 2023.05

* **论文链接：** [Landslide susceptibility modeling by interpretable neural network](https://www.nature.com/articles/s43247-023-00806-5)

### **5. [利用可解释性 AI ，分析澳大利亚吉普斯兰市的不同地理因素](https://hyper.ai/news/33994)**
* **中文解读：** [https://hyper.ai/news/33994](https://hyper.ai/news/33994)

* **科研团队：** 澳大利亚国立大学、悉尼科技大学的研究团队

* **相关研究：** 随机森林模型、机器学习模型、交叉验证技术。XAI可以根据地理特征对野火发生进行有效预测

* **发布期刊：** ScienceDirect, 2023.06

* **论文链接：** [Explainable artificial intelligence (XAI) for interpreting the contributing factors feed into the wildfire susceptibility prediction model](https://www.sciencedirect.com/science/article/pii/S0048969723016224)

### **6. [基于机器学习的洪水预报模型](https://hyper.ai/news/31060)**
* **中文解读：** [https://hyper.ai/news/31060](https://hyper.ai/news/31060)

* **科研团队：** 谷歌研究团队

* **相关研究：** HydroATLAS project、长短期记忆网络LSTM、编码器-解码器、交叉验证、性能优于最先进 GloFAS 预报模型

* **发布期刊：** nature, 2024.03

* **论文链接：** [Global prediction of extreme floods in ungauged watersheds](https://www.nature.com/articles/s41586-024-07145-1)

### **7. [ED-DLSTM实现无监测数据地区洪水预测](https://hyper.ai/news/32138)**
* **中文解读：** [https://hyper.ai/news/32138](https://hyper.ai/news/32138)

* **科研团队：** 中国科学院成都山地灾害与环境研究所欧阳朝军团队

* **相关研究：** 2 千个水文站数据、训练数据集来自美国、英国、中欧、加拿大、跨区域时空集成模型、编码器-解码器、多模态数据、空间静态网格属性数据、残差卷积、

* **发布期刊：** The Innovation, 2024.04

* **论文链接：** [Deep learning for cross-region streamflow and flood forecasting at a global scale](https://doi.org/10.1016/j.xinn.2024.100617)
### **8. [ChloroFormer 模型提前预警海洋藻类爆发](https://hyper.ai/news/34544)**
* **中文解读：** [https://hyper.ai/news/34544](https://hyper.ai/news/34544)

* **科研团队：** 浙江大学 GIS 实验室

* **相关研究：** TZ02 数据集、深度学习模型 ChloroFormer、Transformer 神经网络、频率滤波器机制、频率注意力机制、ChloroFormer 在叶绿素 a 的短期和中期预测上，都超越了基线
* **发布期刊：** Water Research, 2024.10

* **论文链接：** [Enhanced forecasting of chlorophyll-a concentration in coastal waters through integration of Fourier analysis and Transformer networks](https://doi.org/10.1016/j.watres.2024.122160 )

## **AI4S 政策解读：AI4S Policy**
### **1. [科技部出台政策防范学术界 AI 枪手](https://hyper.ai/news/29228)**
* **中文解读：** [https://hyper.ai/news/29228](https://hyper.ai/news/29228)

* **发布时间：** 2023.12

* **详情链接：** [科技部监督司发布《负责任研究行为规范指引（2023）》](https://www.most.gov.cn/kjbgz/202312/t20231221_189240.html 
)

### **2. [政策：科技部会同自然科学基金委启动「人工智能驱动的科学研究」( AI for Science ) 专项部署工作](https://hyper.ai/news/34017)**
* **中文解读：** [https://hyper.ai/news/34017](https://hyper.ai/news/34017)

* **发布时间：** 2023.03

* **详情链接：** [科技部启动“人工智能驱动的科学研究”专项部署工作](http://www.news.cn/2023-03/27/c_1129468666.htm)


## **其他：Others**

### **1. [TacticAI 足球助手战术布局实用性高达 90%](https://hyper.ai/news/30454)**
* **中文解读：** [https://hyper.ai/news/30454](https://hyper.ai/news/30454)

* **科研团队：** 谷歌 DeepMind 与利物浦足球俱乐部

* **相关研究：** Geometric deep learning、GNN、predictive model、generative model。射球机会提升 13%

* **发布期刊：** Nature, 2024.03

* **论文链接：** [TacticAI: an AI assistant for football tactics](https://www.nature.com/articles/s41467-024-45965-x 
)

### **2. [去噪扩散模型 SPDiff 实现长程人流移动模拟](https://hyper.ai/news/30069)**
* **中文解读：** [https://hyper.ai/news/30069](https://hyper.ai/news/30069)

* **科研团队：** 清华大学电子工程系城市科学与计算研究中心、清华大学深圳国际研究生院深圳市泛在数据赋能重点实验室、鹏城实验室的研究团队

* **相关研究：** GC 数据集、UCY 数据集、条件去噪扩散模型、SPDiff、GN、EGCL、LSTM、多帧推演训练算法。5% 训练数据量即可达到最优性能

* **发布期刊：** Nature, 2024.02

* **论文链接：** [Social Physics Informed Diffusion Model for Crowd Simulation](https://arxiv.org/abs/2402.06680)

### **3. [智能化科学设施推进科研范式变革](https://hyper.ai/news/29570)**
* **中文解读：** [https://hyper.ai/news/29570](https://hyper.ai/news/29570)

* **科研团队：** 上海交通大学梅宏研究团队

*  **相关研究：** 科学领域大模型、生成式模拟与反演、自主智能无人实验、大规模可信科研协作、AI 科研助手

* **发布期刊：** 中国科学院院刊，2023.12

* **论文链接：** [AI for Science：智能化科学设施变革基础研究](http://www.bulletin.cas.cn/previewFile?id=52965146&type=pdf&lang=zh)

### **4. [DeepSymNet 基于监督学习来表示符号表达式](https://hyper.ai/news/29243)**
* **中文解读：** [https://hyper.ai/news/29243](https://hyper.ai/news/29243)

* **科研团队：** 中国科学院半导体研究所吴敏研究团队

* **相关研究：** [符号网络数据集](https://hyper.ai/datasets/29321)、DSNOrg、DSNB、DSNBM、监督学习。使用标签更短、减少预测的搜索空间、提升算法鲁棒性

* **发布期刊：** Journals & Magazines, 2023.11

* **论文链接：** [Discovering Mathematical Expressions Through DeepSymNet: A Classification-Based Symbolic Regression Framework](https://ieeexplore.ieee.org/document/10327762)

### **5. [大语言模型 ChipNeMo 辅助工程师完成芯片设计](https://hyper.ai/news/29134)**
* **中文解读：** [https://hyper.ai/news/29134](https://hyper.ai/news/29134)

* **科研团队：** 英伟达研究团队

* **相关研究：** 领域自适应技术、NVIDIA NeMo、domain-adapted retrieval models、RAG、supervised fine-tuning with domain-specific instructions、DAPT、SFT、Tevatron、LLM

* **发布期刊：** arXiv, 2024.04

* **论文链接：** [ChipNeMo: Domain-Adapted LLMs for Chip Design](https://arxiv.org/abs/2311.00176)

### **6. [AlphaGeometry 可解决几何学问题](https://hyper.ai/news/29059)**
* **中文解读：** [https://hyper.ai/news/29059](https://hyper.ai/news/29059)

* **科研团队：** 谷歌 DeepMind 研究团队

* **相关研究：** neural language model、symbolic deduction engine、语言模型

* **发布期刊：** Nature, 2024.01

* **论文链接：** [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)

### **7. [强化学习用于城市空间规划](https://hyper.ai/news/28917)**
* **中文解读：** [https://hyper.ai/news/28917](https://hyper.ai/news/28917)

* **科研团队：** 清华大学李勇研究团队

* **相关研究：** 深度强化学习、human–artificial intelligence collaborative 框架、城市规划模型、策略网络、价值网络、GNN。在服务和生态指标上击败了 8 名专业人类规划师

* **发布期刊：** Nature Computational Science, 2023.09

* **论文链接：** [Spatial planning of urban communities via deep reinforcement learning](https://www.nature.com/articles/s43588-023-00503-5)

### **8. [ChatArena 框架，与大语言模型一起玩狼人杀](https://hyper.ai/news/28576)**
* **中文解读：** [https://hyper.ai/news/28576](https://hyper.ai/news/28576)

* **科研团队：** 清华大学李鹏研究团队

* **相关研究：** 非参数学习机制、语言模型、Prompt

* **发布期刊：** arxiv, 2023.09

* **论文链接：** [Exploring Large Language Models for Communication Games:
An Empirical Study on Werewolf](https://arxiv.org/pdf/2309.04658.pdf)

### **9. [综述：30 位学者合力发表 Nature，10 年回顾解构 AI 如何重塑科研范式](https://hyper.ai/news/28166)**
* **中文解读：** [https://hyper.ai/news/28166](https://hyper.ai/news/28166)

* **主要内容：** 来自斯坦福大学计算机科学与基因技术学院的博士后 Hanchen Wang，与佐治亚理工学院计算科学与工程专业的 Tianfan Fu，以及康奈尔大学计算机系的 Yuanqi Du 等 30 人，回顾了过去十年间，基础科研领域中的 AI 角色，并提出了仍然存在的挑战和不足

* **论文链接：** [Scientific discovery in the age of artificial intelligence](https://www.nature.com/articles/s41586-023-06221-2)

### **10. [Ithaca 协助金石学家进行文本修复、时间归因和地域归因的工作](https://hyper.ai/news/28140)**
* **中文解读：** [https://hyper.ai/news/28140](https://hyper.ai/news/28140)

* **科研团队：** DeepMind 和威尼斯福斯卡里大学的研究团队

* **相关研究：** I.PHI 数据集、Ithaca 模型、Kullback-Leibler 散度、交叉熵损失函数。文本修复工作的准确率达到 62%，时间归因误差在 30 年内，地域归因准确率达到 71%

* **发布期刊：** Nature, 2020.03

* **论文链接：** [Restoring and attributing ancient texts using deep neural networks](https://www.nature.com/articles/s41586-022-04448-z)

### **11. [AI 在超光学中的正问题及逆问题、基于超表面系统的数据分析](https://hyper.ai/news/34006)**
* **中文解读：** [https://hyper.ai/news/34006](https://hyper.ai/news/34006)

* **科研团队：** 香港城市大学的研究团队

* **相关研究：** Predicting NN、深度神经网络。预测准确率达到 99% 以上

* **发布期刊：** ACS Publications, 2022.06

* **论文链接：** [Artificial Intelligence in Meta-optics](https://pubs.acs.org/doi/10.1021/acs.chemrev.2c00012)

### **12. [一种新的地理空间人工智能方法：地理神经网络加权逻辑回归](https://hyper.ai/news/30608)**
* **中文解读：** [https://hyper.ai/news/30608](https://hyper.ai/news/30608)

* **科研团队：** 浙江大学杜震洪研究团队

* **相关研究：** 空间模式、神经网络、Shapley 加性解释、反距离加权插值、二元交叉熵损失函数、五折交叉验证、在矿产资源预测评价方面优于其他先进模型

* **发布期刊：** International Journal of Applied Earth Observation and Geoinformation, 2024.04

* **论文链接：** [Enhancing mineral prospectivity mapping with geospatial artificial intelligence: A geographically neural network-weighted logistic regression approach](https://doi.org/10.1016/j.jag.2024.103746)

### **13. [利用扩散模型生成神经网络参数，将时空少样本学习转变为扩散模型的预训练问题](https://hyper.ai/news/30545)**
* **中文解读：** [https://hyper.ai/news/30545](https://hyper.ai/news/30545)

* **科研团队：** 清华大学电子工程系城市科学与计算研究中心李勇研究团队

* **相关研究：** 智慧城市、时空数据、知识迁移、MetaLA、PEMS-BAy、Transformer 扩散模型、条件生成框架 GPD、神经网络、神经网络参数、预训练 + 提示微调

* **发布期刊：** ICLR 2024, 2024.01

* **论文链接：** [Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation](https://openreview.net/forum?id=QyFm3D3Tzi)

### **14. [李飞飞团队 AI4S 最新洞察：16 项创新技术汇总，覆盖生物/材料/医疗/问诊](https://hyper.ai/news/31499)**
* **中文解读：** [https://hyper.ai/news/31499](https://hyper.ai/news/31499)

* **主要内容：** 斯坦福大学 HAI 研究中心发布《2024 年人工智能指数报告》。这份报告全面追踪了 2023 年全球人工智能的发展趋势。还探讨人工智能在科学和医学领域的深远影响。报告中展示了 2023 年 AI 在科学领域的辉煌成就，以及 AI 在医疗领域取得的重要创新成果，包括 SynthSR 和 ImmunoSEIRA 等突破性技术。此外，还分析了 FDA 对 AI 医疗设备审批的趋势，为行业提供了宝贵的参考。

### **15. [精准预测武汉房价！osp-GNNWR 模型准确描述复杂空间过程和地理现象](https://hyper.ai/news/32453)**
* **中文解读：** [https://hyper.ai/news/32453](https://hyper.ai/news/32453)

* **科研团队：** 浙大 GIS 实验室吴森森团队

* **相关研究：** 神经网络、空间邻近性度量、地理神经网络加权回归方法、安居客 968 个不同房地产样本的数据集、空间回归模型、梯度下降算法

* **发布期刊：** International Journal of Geographical Information Science, 2024.04

* **论文链接：** [A neural network model to optimize the measure of spatial proximity in geographically weighted regression approach: a case study on house price in Wuhan](https://www.tandfonline.com/doi/abs/10.1080/13658816.2024.2343771)

### **16. [首个海洋大语言模型 OceanGPT 入选 ACL 2024！水下具身智能成现实](https://hyper.ai/news/33044)**
* **中文解读：** [https://hyper.ai/news/33044](https://hyper.ai/news/33044)

* **科研团队：** 浙江大学计算机科学与技术学院张宁豫、陈华钧团队

* **相关研究：** 海洋领域大语言模型、正则表达式、哈希算法海洋科学指令生成框架 DoInstruct、多 Agent 协作、gpt-3.5-turbo、BM25 算法、LLaMA-2、Vicuna-7b-1.5、具身智能

* **发布期刊：** ACL 2024, 2024.05

* **论文链接：** [OceanGPT: A Large Language Model for Ocean Science Tasks](https://arxiv.org/abs/2310.02031)

### **17. [引入零样本学习，发布针对甲骨文破译优化的条件扩散模型](https://hyper.ai/news/33010)**
* **中文解读：** [https://hyper.ai/news/33010](https://hyper.ai/news/33010)

* **科研团队：** 华中科技大学白翔、刘禹良研究团队联合阿德莱德大学、安阳师范学院、华南理工大学团队

* **相关研究：** 条件扩散模型、图像生成技术、局部分析采样技术、HUST-OBS 数据集、EVOBC 数据集、ResNet-101 骨干网络、OCR 技术、零样本学习策略、风格编码器、内容编码器

* **发布期刊：** ACL 2024, 2024.06

* **论文链接：** [Deciphering Oracle Bone Language with Diffusion Models](https://doi.org/10.48550/arXiv.2406.00684)

### **18. [斯坦福/苹果等 23 所机构发布 DCLM 基准测试，基础模型与 Llama3 8B 表现相当](https://hyper.ai/news/33001)**
* **中文解读：** [https://hyper.ai/news/33001](https://hyper.ai/news/33001)

* **科研团队：** 华盛顿大学、斯坦福大学、苹果等 23 所机构联手

* **相关研究：** 语言模型、DCLM 基准测试、Transformer、MMLU

* **发布期刊：** arXiv, 2024.06

* **论文链接：** [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)

### **19. [PoCo 解决数据源异构难题，实现机器人多任务灵活执行](https://hyper.ai/news/32765)**
* **中文解读：** [https://hyper.ai/news/32765](https://hyper.ai/news/32765)

* **科研团队：** 麻省理工研究人员

* **相关研究：** 去噪扩散概率模型、去噪扩散隐式模型、扩散模型的概率合成、机器人策略组合框架 PoCo

* **发布期刊：** arXiv, 2024.05

* **论文链接：** [PoCo: Policy Composition from and for Heterogeneous Robot Learning](https://arxiv.org/abs/2402.02511)

### **20. [含 14 万张图像！甲骨文数据集助力团队摘冠 ACL 最佳论文](https://hyper.ai/news/33826)**
* **中文解读：** [https://hyper.ai/news/33826](https://hyper.ai/news/33826)

* **科研团队：** 华中科技大学白翔教授研究团队

* **相关研究：** HUST-OBC 数据集、无监督的视觉对比学习模型

* **发布期刊：** Scientific Data, 2024.06

* **论文链接：** [An open dataset for oracle bone script recognition and decipherment](https://arxiv.org/abs/2401.15365)

### **21. [用机器学习分离抹香鲸发音字母表，高度类似人类语言，信息承载能力更强](https://hyper.ai/news/33433)**
* **中文解读：** [https://hyper.ai/news/33433](https://hyper.ai/news/33433)

* **科研团队：** 麻省理工学院 Pratyusha Sharma 以及 CETI 的研究团队

* **相关研究：**  DSWP 数据集、机器学习、抹香鲸声音具有结构性

* **发布期刊：** Nature Communications, 2024.05

* **论文链接：** [Contextual and combinatorial structure in sperm whale vocalisations](https://www.nature.com/articles/s41467-024-47221-8)

### **22. [基于预训练 LLM 提出信道预测方案，GPT-2 赋能无线通信物理层](https://hyper.ai/news/33195)**
* **中文解读：** [https://hyper.ai/news/33195](https://hyper.ai/news/33195)

* **科研团队：** 北京大学电子学院程翔团队

* **相关研究：**  QuaDRiGa 仿真器、大语言模型、信道预测神经网络、预处理模块、嵌入模块、预训练 LLM 模块、输出模块

* **发布期刊：** Journal of Communications and Information Networks, 2024.06

* **论文链接：** [LLM4CP: Adapting Large Language Models for Channel Prediction](https://ieeexplore.ieee.org/document/10582829)

### **23. [首个多缝线刺绣生成对抗网络模型](https://hyper.ai/news/34669)**
* **中文解读：** [https://hyper.ai/news/34669](https://hyper.ai/news/34669)

* **科研团队：** 复武汉纺织大学计算机与人工智能学院可视计算与数字纺织团队

* **相关研究：**  多针刺绣数据集、生成对抗网络模型、卷积神经网络、CNN、多缝线刺绣生成对抗网络模型 MSEmbGAN、区域感知纹理生成网络、着色网络、可提高刺绣中纹理真实度和色彩保真度等关键方面的精度

* **发布期刊：** IEEE Transactions on Visualization and Computer Graphics, 2024

* **论文链接：** [MSEmbGAN: Multi-Stitch Embroidery Synthesis via Region-Aware Texture Generation](https://csai.wtu.edu.cn/TVCG01/index.html)

