# Awesome AI for Science
**EN** | [CN](README_CN.md)
- [**Foreword**](#foreword)
- [**AI+ Biopharmaceutical**](#ai-biopharmaceutical)
  - [**1. AdaDR outperforms multiple benchmark methods in drug repositioning**](#1-adadr-outperforms-multiple-benchmark-methods-in-drug-repositioning)
  - [**2. IMN4NPD accelerates the dereplication of extensive clusters in molecular networks, providing annotations for self-loops and paired nodes**](#2-imn4npd-accelerates-the-dereplication-of-extensive-clusters-in-molecular-networks-providing-annotations-for-self-loops-and-paired-nodes)
  - [**3. Deep generative model MIDAS for mosaic integration of single-cell multi-omics data**](#3-deep-generative-model-midas-for-mosaic-integration-of-single-cell-multi-omics-data)
  - [**4. ResGen: A 3D molecular generation model based on protein pockets**](#4-resgen-a-3d-molecular-generation-model-based-on-protein-pockets)
  - [**5. Large models + machine learning for high-precision prediction of enzyme kinetic parameters**](#5-large-models--machine-learning-for-high-precision-prediction-of-enzyme-kinetic-parameters)
  - [**6. MIT uses deep learning to discover novel antibiotics**](#6-mit-uses-deep-learning-to-discover-novel-antibiotics)
  - [**7. Neural networks decipher GPCR-G protein coupling selectivity**](#7-neural-networks-decipher-gpcr-g-protein-coupling-selectivity)
  - [**8. Macformer macrocyclizes the acyclic drug fedratinib**](#8-macformer-macrocyclizes-the-acyclic-drug-fedratinib)
  - [**9. Regression network + CGMD predicts self-assembly properties of tens of billions of peptides**](#9-regression-network--cgmd-predicts-self-assembly-properties-of-tens-of-billions-of-peptides)
  - [**10. Unsupervised learning predicts 71 million gene mutations**](#10-unsupervised-learning-predicts-71-million-gene-mutations)
  - [**11. Odor analysis AI developed based on Graph Neural Networks (GNN)**](#11-odor-analysis-ai-developed-based-on-graph-neural-networks-gnn)
  - [**12. Graph neural networks screen for safe and highly effective anti-aging ingredients**](#12-graph-neural-networks-screen-for-safe-and-highly-effective-anti-aging-ingredients)
  - [**13. Machine learning quantitatively analyzes dopamine release amount and location**](#13-machine-learning-quantitatively-analyzes-dopamine-release-amount-and-location)
  - [**14. Machine learning discovers three anti-aging drugs**](#14-machine-learning-discovers-three-anti-aging-drugs)
  - [**15. Deep learning screens for novel antibiotics inhibiting Acinetobacter baumannii**](#15-deep-learning-screens-for-novel-antibiotics-inhibiting-acinetobacter-baumannii)
  - [**16. Machine learning models applied to predict bioink printability**](#16-machine-learning-models-applied-to-predict-bioink-printability)
  - [**17. Machine learning differentiates pluripotent stem cells**](#17-machine-learning-differentiates-pluripotent-stem-cells)
  - [**18. Machine learning model predicts drug release rate of long-acting injectables**](#18-machine-learning-model-predicts-drug-release-rate-of-long-acting-injectables)
  - [**19. Machine learning algorithm effectively predicts plant antimalarial properties**](#19-machine-learning-algorithm-effectively-predicts-plant-antimalarial-properties)
  - [**20. Machine learning ensemble method predicts immunogenicity of viral protein fragments**](#20-machine-learning-ensemble-method-predicts-immunogenicity-of-viral-protein-fragments)
  - [**21. Generative AI used to develop novel antibiotics**](#21-generative-ai-used-to-develop-novel-antibiotics)
  - [**22. Deep learning-based automated, high-speed, multidimensional single-particle tracking system**](#22-deep-learning-based-automated-high-speed-multidimensional-single-particle-tracking-system)
  - [**23. ProEnsemble machine learning framework: Optimizing evolutionary pathway promoter combinations**](#23-proensemble-machine-learning-framework-optimizing-evolutionary-pathway-promoter-combinations)
  - [**24. Microenvironment-aware graph neural network ProtLGN guides directed protein evolution**](#24-microenvironment-aware-graph-neural-network-protlgn-guides-directed-protein-evolution)
  - [**25. Deep learning model AlphaPPIMd: Exploring conformational ensembles of protein-protein complexes**](#25-deep-learning-model-alphappimd-exploring-conformational-ensembles-of-protein-protein-complexes)
  - [**26. Novel tumor-suppressor protein degrader dp53m inhibits cancer cell proliferation**](#26-novel-tumor-suppressor-protein-degrader-dp53m-inhibits-cancer-cell-proliferation)
  - [**27. CVPR Best Student Paper! Multimodal model BioCLIP achieves zero-shot learning**](#27-cvpr-best-student-paper-multimodal-model-bioclip-achieves-zero-shot-learning)
  - [**28. 100 million parameters! Cell foundation model scFoundation models 20,000 genes simultaneously**](#28-100-million-parameters-cell-foundation-model-scfoundation-models-20000-genes-simultaneously)
  - [**29. Accepted by ICML, protein language model ESM-AA surpasses traditional SOTA**](#29-accepted-by-icml-protein-language-model-esm-aa-surpasses-traditional-sota)
  - [**30. SPACE algorithm published in Cell sub-journal! Tissue module discovery capabilities lead similar tools**](#30-space-algorithm-published-in-cell-sub-journal-tissue-module-discovery-capabilities-lead-similar-tools)
  - [**31. New breakthroughs based on AlphaFold reveal dynamic protein diversity**](#31-new-breakthroughs-based-on-alphafold-reveal-dynamic-protein-diversity)
  - [**32. P450Diffusion: De novo design method for P450 enzymes developed based on diffusion models**](#32-p450diffusion-de-novo-design-method-for-p450-enzymes-developed-based-on-diffusion-models)
  - [**33. Equivariant graph neural networks used for target protein binding site prediction, boosting performance by 20%**](#33-equivariant-graph-neural-networks-used-for-target-protein-binding-site-prediction-boosting-performance-by-20)
  - [**34. 20 experimental data points create an AI protein milestone! FSFP effectively optimizes protein pre-training models**](#34-20-experimental-data-points-create-an-ai-protein-milestone-fsfp-effectively-optimizes-protein-pre-training-models)
  - [**35. Transferable deep learning model identifies multiple types of RNA modifications, significantly reducing computational costs**](#35-transferable-deep-learning-model-identifies-multiple-types-of-rna-modifications-significantly-reducing-computational-costs)
  - [**36. InstructProtein: Aligning protein language with human language using knowledge instructions**](#36-instructprotein-aligning-protein-language-with-human-language-using-knowledge-instructions)
  - [**37. Protein-to-text generation framework ProtT3 enables cross-modal interpretation of protein data and text information**](#37-protein-to-text-generation-framework-prott3-enables-cross-modal-interpretation-of-protein-data-and-text-information)
  - [**38. CPDiffusion model designs functional proteins fully automatically at an ultra-low cost**](#38-cpdiffusion-model-designs-functional-proteins-fully-automatically-at-an-ultra-low-cost)
  - [**39. A novel protein homolog detection method based on protein language models and dense retrieval techniques**](#39-a-novel-protein-homolog-detection-method-based-on-protein-language-models-and-dense-retrieval-techniques)
  - [**40. AlphaProteo efficiently designs target protein binders, increasing affinity by 300 times**](#40-alphaproteo-efficiently-designs-target-protein-binders-increasing-affinity-by-300-times)
  - [**41. Novel denoising protein language model DePLM outperforms SOTA models in mutation effect prediction**](#41-novel-denoising-protein-language-model-deplm-outperforms-sota-models-in-mutation-effect-prediction)
  - [**42. Geometric deep generative model DynamicBind enables dynamic protein docking prediction**](#42-geometric-deep-generative-model-dynamicbind-enables-dynamic-protein-docking-prediction)
  - [**43. Drug discovery large language model Y-Mol completely outperforms LLaMA2**](#43-drug-discovery-large-language-model-y-mol-completely-outperforms-llama2)
  - [**44. Universal molecular inverse folding model UniIF further complements AlphaFold 3**](#44-universal-molecular-inverse-folding-model-uniif-further-complements-alphafold-3)
  - [**45. Pre-trained protein language model ProSST integrates protein structure information more effectively**](#45-pre-trained-protein-language-model-prosst-integrates-protein-structure-information-more-effectively)
  - [**46. Macrocyclic peptide binder framework RFpeptides offers new possibilities for undruggable proteins**](#46-macrocyclic-peptide-binder-framework-rfpeptides-offers-new-possibilities-for-undruggable-proteins)
  - [**47. Genome foundation model Evo enables prediction and generation from molecular to genome scales**](#47-genome-foundation-model-evo-enables-prediction-and-generation-from-molecular-to-genome-scales)
  - [**48. DigFrag accurately segments molecular fragments using AI and generates 44 drug/pesticide molecules**](#48-digfrag-accurately-segments-molecular-fragments-using-ai-and-generates-44-drugpesticide-molecules)
  - [**49. Protein sequence large language model pre-training method PRIME**](#49-protein-sequence-large-language-model-pre-training-method-prime)
  - [**50. Self-supervised deep learning method revolutionizes 3D reconstruction in cryo-electron microscopy**](#50-self-supervised-deep-learning-method-revolutionizes-3d-reconstruction-in-cryo-electron-microscopy)
  - [**51. Multimodal protein generation method PLAID generates sequences and all-atom protein structures simultaneously**](#51-multimodal-protein-generation-method-plaid-generates-sequences-and-all-atom-protein-structures-simultaneously)
  - [**52. Targeted molecular optimization method MOLRL based on latent reinforcement learning**](#52-targeted-molecular-optimization-method-molrl-based-on-latent-reinforcement-learning)
  - [**53. Viral variation driver prediction framework E2VD predicts evolutionary directions for COVID-19/HIV/Influenza viruses**](#53-viral-variation-driver-prediction-framework-e2vd-predicts-evolutionary-directions-for-covid-19hivinfluenza-viruses)
  - [**54. Medical language model MedFound approaches expert physician reasoning capabilities**](#54-medical-language-model-medfound-approaches-expert-physician-reasoning-capabilities)
  - [**55. 4D diffusion model AlphaFolding fills the gap in dynamic protein structure prediction**](#55-4d-diffusion-model-alphafolding-fills-the-gap-in-dynamic-protein-structure-prediction)
  - [**56. PepPrCLIP pipeline for designing short proteins holds promise for developing new cancer therapies**](#56-pepprclip-pipeline-for-designing-short-proteins-holds-promise-for-developing-new-cancer-therapies)
  - [**57. Boltzmann alignment technique drastically improves protein binding free energy prediction efficacy**](#57-boltzmann-alignment-technique-drastically-improves-protein-binding-free-energy-prediction-efficacy)
  - [**58. Novel large-scale flow-based protein backbone generator Proteina achieves SOTA in de novo protein backbone design**](#58-novel-large-scale-flow-based-protein-backbone-generator-proteina-achieves-sota-in-de-novo-protein-backbone-design)
  - [**59. UniGEM model achieves synergistic enhancement of two tasks based on diffusion models for the first time**](#59-unigem-model-achieves-synergistic-enhancement-of-two-tasks-based-on-diffusion-models-for-the-first-time)
  - [**60. RFdiffusion evolves further, realizing atomic-accuracy de novo antibody design**](#60-rfdiffusion-evolves-further-realizing-atomic-accuracy-de-novo-antibody-design)
  - [**61. First protein-RNA language model fusion scheme sets new SOTA in binding affinity prediction**](#61-first-protein-rna-language-model-fusion-scheme-sets-new-sota-in-binding-affinity-prediction)
  - [**62. Virtual tissue model Celcomen achieves causal inference identifiability in spatial transcriptomics analysis for the first time**](#62-virtual-tissue-model-celcomen-achieves-causal-inference-identifiability-in-spatial-transcriptomics-analysis-for-the-first-time)
  - [**63. AlphaFold-Metainference method accurately predicts disordered protein structural ensembles**](#63-alphafold-metainference-method-accurately-predicts-disordered-protein-structural-ensembles)
  - [**64. High-accuracy RNA structure prediction framework DRfold2 surpasses SOTA in multiple benchmarks**](#64-high-accuracy-rna-structure-prediction-framework-drfold2-surpasses-sota-in-multiple-benchmarks)
  - [**65. New protein design algorithm DRAKES breaks through the biological sequence design bottleneck**](#65-new-protein-design-algorithm-drakes-breaks-through-the-biological-sequence-design-bottleneck)
  - [**66. Machine learning-assisted UV absorbance spectroscopy for detecting microbial contamination**](#66-machine-learning-assisted-uv-absorbance-spectroscopy-for-detecting-microbial-contamination)
  - [**67. Utilizing protein sequence generative models for overlapping gene design**](#67-utilizing-protein-sequence-generative-models-for-overlapping-gene-design)
  - [**68. Prediction framework PUPS enables single-cell level protein subcellular localization**](#68-prediction-framework-pups-enables-single-cell-level-protein-subcellular-localization)
  - [**69. UniMoMo: The first unified generative framework across molecular species enables multi-type drug molecular design**](#69-unimomo-the-first-unified-generative-framework-across-molecular-species-enables-multi-type-drug-molecular-design)
  - [**70. Protein language model Prot42 generates high-affinity binders using only the target protein sequence**](#70-protein-language-model-prot42-generates-high-affinity-binders-using-only-the-target-protein-sequence)
  - [**71. Unified biomolecular dynamics simulator UniSim achieves unified time-coarsened dynamics simulation across molecular types and chemical environments for the first time**](#71-unified-biomolecular-dynamics-simulator-unisim-achieves-unified-time-coarsened-dynamics-simulation-across-molecular-types-and-chemical-environments-for-the-first-time)
  - [**72. Computational biology algorithm SimplifiedBondfinder uncovers 69 novel nitrogen-oxygen-sulfur bonds**](#72-computational-biology-algorithm-simplifiedbondfinder-uncovers-69-novel-nitrogen-oxygen-sulfur-bonds)
  - [**73. Novel protein sequence design method FAMPNN simultaneously processes protein backbone and sidechain information**](#73-novel-protein-sequence-design-method-fampnn-simultaneously-processes-protein-backbone-and-sidechain-information)
  - [**74. Atomistic protein design method La-Proteina generates proteins with up to 800 residues at high precision**](#74-atomistic-protein-design-method-la-proteina-generates-proteins-with-up-to-800-residues-at-high-precision)
  - [**75. APM model specifically designed for multi-chain protein complexes enables all-atom design and functional optimization**](#75-apm-model-specifically-designed-for-multi-chain-protein-complexes-enables-all-atom-design-and-functional-optimization)
  - [**76. New intrinsically disordered region-binding protein design method Logos specializes in undruggable targets**](#76-new-intrinsically-disordered-region-binding-protein-design-method-logos-specializes-in-undruggable-targets)
  - [**77. Novel protein dynamic fusion representation framework FusionProt released, enabling iterative information exchange**](#77-novel-protein-dynamic-fusion-representation-framework-fusionprot-released-enabling-iterative-information-exchange)
  - [**78. Transcriptome-guided diffusion model MorphDiff released to accelerate phenotypic drug discovery**](#78-transcriptome-guided-diffusion-model-morphdiff-released-to-accelerate-phenotypic-drug-discovery)
  - [**79. AlphaPPIMI framework significantly enhances generalization, surpassing existing methods in PPI interface modulator prediction**](#79-alphappimi-framework-significantly-enhances-generalization-surpassing-existing-methods-in-ppi-interface-modulator-prediction)
  - [**80. A novel fusion neural network framework efficiently predicts multi-metal binding sites in protein sequences**](#80-a-novel-fusion-neural-network-framework-efficiently-predicts-multi-metal-binding-sites-in-protein-sequences)
  - [**81. Highly synthesizable molecular projection framework ReaSyn released, achieving ultra-high reconstruction rates and pathway diversity**](#81-highly-synthesizable-molecular-projection-framework-reasyn-released-achieving-ultra-high-reconstruction-rates-and-pathway-diversity)
  - [**82. Constrained reinforcement learning framework Ctrl-DNA released, realizing "targeted control" of specific cell gene expression**](#82-constrained-reinforcement-learning-framework-ctrl-dna-released-realizing-targeted-control-of-specific-cell-gene-expression)
  - [**83. PLACER framework resolves the atomic-level modeling challenge of protein conformational heterogeneity**](#83-placer-framework-resolves-the-atomic-level-modeling-challenge-of-protein-conformational-heterogeneity)
  - [**84. Squidiff enables multi-scenario transcriptome simulation, boosting precision medicine and spatial medicine development**](#84-squidiff-enables-multi-scenario-transcriptome-simulation-boosting-precision-medicine-and-spatial-medicine-development)
  - [**85. Generative model PepTron and new evaluation benchmark released, reshaping prediction capabilities for disordered protein ensembles**](#85-generative-model-peptron-and-new-evaluation-benchmark-released-reshaping-prediction-capabilities-for-disordered-protein-ensembles)
  - [**86. MIT and Harvard propose end-to-end AI workflow CleaveNet to overcome highly specific protease substrate design challenges**](#86-mit-and-harvard-propose-end-to-end-ai-workflow-cleavenet-to-overcome-highly-specific-protease-substrate-design-challenges)
  - [**87. Goethe University Frankfurt team proposes a multi-scale classification framework to decode the complexity of the human E3 ligome**](#87-goethe-university-frankfurt-team-proposes-a-multi-scale-classification-framework-to-decode-the-complexity-of-the-human-e3-ligome)
  - [**88. Basecamp and NVIDIA jointly release the EDEN foundation model, enabling AI-programmable therapeutic design**](#88-basecamp-and-nvidia-jointly-release-the-eden-foundation-model-enabling-ai-programmable-therapeutic-design)
  - [**89. Microsoft and others propose the multimodal AI framework GigaTIME to generate virtual mIF atlases from routine pathology slides**](#89-microsoft-and-others-propose-the-multimodal-ai-framework-gigatime-to-generate-virtual-mif-atlases-from-routine-pathology-slides)
  - [**90. MIT proposes deep learning language model Pichia-CLM to optimize codons for enhanced recombinant protein yield**](#90-mit-proposes-deep-learning-language-model-pichia-clm-to-optimize-codons-for-enhanced-recombinant-protein-yield)
  - [**91. MIT and ETH jointly propose deep learning framework APOLLO to efficiently integrate and disentangle single-cell multimodal data**](#91-mit-and-eth-jointly-propose-deep-learning-framework-apollo-to-efficiently-integrate-and-disentangle-single-cell-multimodal-data)
  - [**92. CUHK and others jointly propose the Bi-TEAM framework for unified cross-scale representation learning of modified peptides**](#92-cuhk-and-others-jointly-propose-the-bi-team-framework-for-unified-cross-scale-representation-learning-of-modified-peptides)
  - [**93. Carnegie Mellon University and others propose AQuaRef for quantum refinement of all-atom protein models**](#93-carnegie-mellon-university-and-others-propose-aquaref-for-quantum-refinement-of-all-atom-protein-models)
  - [**94. NVIDIA and others jointly propose the Complexa framework to unify protein binder generation and optimization**](#94-nvidia-and-others-jointly-propose-the-complexa-framework-to-unify-protein-binder-generation-and-optimization)
  - [**95. MIT and CMU jointly propose VibeGen, introducing vibrational dynamics to empower de novo protein design**](#95-mit-and-cmu-jointly-propose-vibegen-introducing-vibrational-dynamics-to-empower-de-novo-protein-design)
  - [**96. Institut Pasteur uses deep learning to predict 2.39 million anti-phage proteins, mapping bacterial immunity**](#96-institut-pasteur-uses-deep-learning-to-predict-239-million-anti-phage-proteins-mapping-bacterial-immunity)
  - [**97. KAIST team utilizes AI to de novo design small-molecule binding proteins, successfully applying them in biosensors**](#97-kaist-team-utilizes-ai-to-de-novo-design-small-molecule-binding-proteins-successfully-applying-them-in-biosensors)
  - [**98. University of Toronto and others propose dnaHNet for efficient hierarchical modeling of genomic sequences**](#98-university-of-toronto-and-others-propose-dnahnet-for-efficient-hierarchical-modeling-of-genomic-sequences)
  - [**99. Queen Mary University of London and others conduct the largest-scale proteogenomic study, revealing molecular disease mechanisms**](#99-queen-mary-university-of-london-and-others-conduct-the-largest-scale-proteogenomic-study-revealing-molecular-disease-mechanisms)
  - [**100. Goethe University Frankfurt and others propose genESOM model: Generative AI breaks through small-sample animal experiments**](#100-goethe-university-frankfurt-and-others-propose-genesom-model-generative-ai-breaks-through-small-sample-animal-experiments)
- [**AI+ Healthcare**](#ai-healthcare)
  - [**1. DeepDR Plus deep learning system predicts diabetic retinopathy using fundus images**](#1-deepdr-plus-deep-learning-system-predicts-diabetic-retinopathy-using-fundus-images)
  - [**2. Logistic regression model analyzes that high green landscape index reduces MetS risk**](#2-logistic-regression-model-analyzes-that-high-green-landscape-index-reduces-mets-risk)
  - [**3. Deep learning system helps junior ophthalmologists increase diagnostic consistency by 12%**](#3-deep-learning-system-helps-junior-ophthalmologists-increase-diagnostic-consistency-by-12)
  - [**4. GSP-GCNs achieve up to 90.2% accuracy in Parkinson's disease diagnosis**](#4-gsp-gcns-achieve-up-to-902-accuracy-in-parkinsons-disease-diagnosis)
  - [**5. Breast cancer prognosis scoring system MIRS**](#5-breast-cancer-prognosis-scoring-system-mirs)
  - [**6. Retinal image foundation model RETFound predicts multiple systemic diseases**](#6-retinal-image-foundation-model-retfound-predicts-multiple-systemic-diseases)
  - [**7. SVM optimizes tactile sensors, braille recognition rate reaches 96.12%**](#7-svm-optimizes-tactile-sensors-braille-recognition-rate-reaches-9612)
  - [**8. CAS Beijing Institute of Genomics establishes an open biomedical imaging archive**](#8-cas-beijing-institute-of-genomics-establishes-an-open-biomedical-imaging-archive)
  - [**9. AI Lunit reads mammograms with accuracy comparable to doctors**](#9-ai-lunit-reads-mammograms-with-accuracy-comparable-to-doctors)
  - [**10. Feature selection strategy detects breast cancer biomarkers**](#10-feature-selection-strategy-detects-breast-cancer-biomarkers)
  - [**11. Gradient boosting machine model accurately predicts BPSD sub-syndrome**](#11-gradient-boosting-machine-model-accurately-predicts-bpsd-sub-syndrome)
  - [**12. Machine learning model predicts patient one-year mortality rate**](#12-machine-learning-model-predicts-patient-one-year-mortality-rate)
  - [**13. New AI brain-computer interface technology allows aphasic patients to "speak"**](#13-new-ai-brain-computer-interface-technology-allows-aphasic-patients-to-speak)
  - [**14. Deep learning-based artificial intelligence detection of pancreatic cancer**](#14-deep-learning-based-artificial-intelligence-detection-of-pancreatic-cancer)
  - [**15. Population effectiveness of machine learning-assisted lung cancer screening**](#15-population-effectiveness-of-machine-learning-assisted-lung-cancer-screening)
  - [**16. Ovarian cancer diagnostic AI fusion model MCF calculates risk using routine lab data and age**](#16-ovarian-cancer-diagnostic-ai-fusion-model-mcf-calculates-risk-using-routine-lab-data-and-age)
  - [**17. Google releases HEAL framework, a 4-step process to assess medical AI tool fairness**](#17-google-releases-heal-framework-a-4-step-process-to-assess-medical-ai-tool-fairness)
  - [**18. Leveraging semantic segmentation to develop spatial transcriptomics semantic annotation tool Pianno**](#18-leveraging-semantic-segmentation-to-develop-spatial-transcriptomics-semantic-annotation-tool-pianno)
  - [**19. AI model UniFMIR breaks the limits of existing fluorescence microscopy imaging**](#19-ai-model-unifmir-breaks-the-limits-of-existing-fluorescence-microscopy-imaging)
  - [**20. Deep learning system improves the accuracy of cancer survival prediction**](#20-deep-learning-system-improves-the-accuracy-of-cancer-survival-prediction)
  - [**21. MemSAM adapts "Segment Anything" model for medical video segmentation**](#21-memsam-adapts-segment-anything-model-for-medical-video-segmentation)
  - [**22. Medical image segmentation model Medical SAM 2 tops the SOTA leaderboard**](#22-medical-image-segmentation-model-medical-sam-2-tops-the-sota-leaderboard)
  - [**23. Machine learning fights chemotherapy resistance and tumor recurrence, building a strong defense against breast cancer stem cells**](#23-machine-learning-fights-chemotherapy-resistance-and-tumor-recurrence-building-a-strong-defense-against-breast-cancer-stem-cells)
  - [**24. Vision-Language model DeepDR-LLM for diabetes care published in Nature sub-journal**](#24-vision-language-model-deepdr-llm-for-diabetes-care-published-in-nature-sub-journal)
  - [**25. Leveling with senior pathologists! Tsinghua team proposes AI foundation model ROAM for precise glioma diagnosis**](#25-leveling-with-senior-pathologists-tsinghua-team-proposes-ai-foundation-model-roam-for-precise-glioma-diagnosis)
  - [**26. Universal medical image segmentation model ScribblePrompt outperforms SAM-based models**](#26-universal-medical-image-segmentation-model-scribbleprompt-outperforms-sam-based-models)
  - [**27. Digital twin brain platform demonstrates critical phenomena and cognitive functions similar to the human brain**](#27-digital-twin-brain-platform-demonstrates-critical-phenomena-and-cognitive-functions-similar-to-the-human-brain)
  - [**28. Automated LLM dialogue Agent simulation system performs initial diagnosis for depression**](#28-automated-llm-dialogue-agent-simulation-system-performs-initial-diagnosis-for-depression)
  - [**29. Deep learning model LucaProt aids in RNA virus identification**](#29-deep-learning-model-lucaprot-aids-in-rna-virus-identification)
  - [**30. Medical image pre-training framework UniMedI breaks down medical data heterogeneity barriers**](#30-medical-image-pre-training-framework-unimedi-breaks-down-medical-data-heterogeneity-barriers)
  - [**31. Multilingual medical large model MMed-Llama 3 better adapts to medical application scenarios**](#31-multilingual-medical-large-model-mmed-llama-3-better-adapts-to-medical-application-scenarios)
  - [**32. Capsule endoscopy image stitching method S2P-Matching assists in image reconstruction**](#32-capsule-endoscopy-image-stitching-method-s2p-matching-assists-in-image-reconstruction)
  - [**33. Multimodal medical benchmark GMAI-MMBench features 284 datasets covering 18 clinical tasks**](#33-multimodal-medical-benchmark-gmai-mmbench-features-284-datasets-covering-18-clinical-tasks)
  - [**34. Novel time series forecasting method CGS-Mask uncovers key indicators for patient survival rates**](#34-novel-time-series-forecasting-method-cgs-mask-uncovers-key-indicators-for-patient-survival-rates)
  - [**35. Non-invasive brain decoding framework fMRI lays the foundation for brain-computer interfaces and cognitive models**](#35-non-invasive-brain-decoding-framework-fmri-lays-the-foundation-for-brain-computer-interfaces-and-cognitive-models)
  - [**36. Medical image segmentation model M2CF-Net improves diagnosis accuracy for Sjogren's syndrome**](#36-medical-image-segmentation-model-m2cf-net-improves-diagnosis-accuracy-for-sjogrens-syndrome)
  - [**37. BSAFusion enables alignment and fusion of multimodal medical images**](#37-bsafusion-enables-alignment-and-fusion-of-multimodal-medical-images)
  - [**38. Multi-Agent LLM framework KG4Diagnosis assists in diagnosing 362 common diseases**](#38-multi-agent-llm-framework-kg4diagnosis-assists-in-diagnosing-362-common-diseases)
  - [**39. Image segmentation model ConDSeg solves soft boundary and co-occurrence issues in medical imaging**](#39-image-segmentation-model-condseg-solves-soft-boundary-and-co-occurrence-issues-in-medical-imaging)
  - [**40. Medical model M³FM enables zero-shot clinical diagnosis, supporting disease reporting and classification**](#40-medical-model-m³fm-enables-zero-shot-clinical-diagnosis-supporting-disease-reporting-and-classification)
  - [**41. Deep learning-based sex estimation from skull CT scans outperforms human forensic experts**](#41-deep-learning-based-sex-estimation-from-skull-ct-scans-outperforms-human-forensic-experts)
  - [**42. AI boosts medical research: Large models become the "golden partner" for training primary care physicians**](#42-ai-boosts-medical-research-large-models-become-the-golden-partner-for-training-primary-care-physicians)
  - [**43. AcneDGNet deep learning algorithm achieves acne lesion detection and grading**](#43-acnedgnet-deep-learning-algorithm-achieves-acne-lesion-detection-and-grading)
  - [**44. Multimodal medical image segmentation model VISTA3D released, achieving 3D image auto-segmentation and interaction**](#44-multimodal-medical-image-segmentation-model-vista3d-released-achieving-3d-image-auto-segmentation-and-interaction)
  - [**45. Multi-plane echocardiography unified segmentation model EchoONE accurately segments multiple planes**](#45-multi-plane-echocardiography-unified-segmentation-model-echoone-accurately-segments-multiple-planes)
  - [**46. Multi-agent dialogue framework simulates medical consultations to aid disease diagnosis**](#46-multi-agent-dialogue-framework-simulates-medical-consultations-to-aid-disease-diagnosis)
  - [**47. Deep learning framework STAIG reveals detailed genetic information in the tumor microenvironment**](#47-deep-learning-framework-staig-reveals-detailed-genetic-information-in-the-tumor-microenvironment)
  - [**48. First all-in-one medical image re-identification framework MaMI reaches SOTA across 11 datasets**](#48-first-all-in-one-medical-image-re-identification-framework-mami-reaches-sota-across-11-datasets)
  - [**49. Many-to-one regression model M2OST accurately predicts gene expression using digital pathology images**](#49-many-to-one-regression-model-m2ost-accurately-predicts-gene-expression-using-digital-pathology-images)
  - [**50. Brain MRI scanning tool MindGlide quantifies multiple sclerosis lesions**](#50-brain-mri-scanning-tool-mindglide-quantifies-multiple-sclerosis-lesions)
  - [**51. Hierarchical distillation multi-instance learning framework HDMIL rapidly processes gigapixel whole-slide images**](#51-hierarchical-distillation-multi-instance-learning-framework-hdmil-rapidly-processes-gigapixel-whole-slide-images)
  - [**52. Universal 3D blood vessel segmentation foundation model vesselFM far exceeds SAM-based models**](#52-universal-3d-blood-vessel-segmentation-foundation-model-vesselfm-far-exceeds-sam-based-models)
  - [**53. Graph neural networks accurately predict lung cancer survival, discovering 3 fatal subtypes**](#53-graph-neural-networks-accurately-predict-lung-cancer-survival-discovering-3-fatal-subtypes)
  - [**54. Fusion strategy AI model predicts septic shock mortality risk**](#54-fusion-strategy-ai-model-predicts-septic-shock-mortality-risk)
  - [**55. World's first clinical Graph-of-Thought model in HIE improves neurocognitive outcome prediction by 15%**](#55-worlds-first-clinical-graph-of-thought-model-in-hie-improves-neurocognitive-outcome-prediction-by-15)
  - [**56. Fine-grained patient cohort modeling using multidimensional EHR data increases length-of-stay prediction accuracy by 16.3%**](#56-fine-grained-patient-cohort-modeling-using-multidimensional-ehr-data-increases-length-of-stay-prediction-accuracy-by-163)
  - [**57. Deep learning model APEX screens potential antibiotic candidates**](#57-deep-learning-model-apex-screens-potential-antibiotic-candidates)
  - [**58. Wastewater epidemiology assessment using gene sequencing and machine learning: ICA-Var method detects viruses up to 4 weeks early**](#58-wastewater-epidemiology-assessment-using-gene-sequencing-and-machine-learning-ica-var-method-detects-viruses-up-to-4-weeks-early)
  - [**59. Bidirectional Brownian bridge diffusion model enhances reproducibility of virtual staining**](#59-bidirectional-brownian-bridge-diffusion-model-enhances-reproducibility-of-virtual-staining)
  - [**60. Medical GraphRAG breaks QA accuracy records, achieving SOTA on 11 benchmark datasets**](#60-medical-graphrag-breaks-qa-accuracy-records-achieving-sota-on-11-benchmark-datasets)
  - [**61. Healthcare Agent automatically detects medical ethics and safety issues**](#61-healthcare-agent-automatically-detects-medical-ethics-and-safety-issues)
  - [**62. Blood cell image classifier CytoDiffusion assists in discovering leukemia, surpassing clinical experts**](#62-blood-cell-image-classifier-cytodiffusion-assists-in-discovering-leukemia-surpassing-clinical-experts)
  - [**63. UCL team proposes federated learning framework MORPHFED for cross-institutional blood morphology analysis**](#63-ucl-team-proposes-federated-learning-framework-morphfed-for-cross-institutional-blood-morphology-analysis)
  - [**64. French team proposes explainable machine learning framework for accurate mortality prediction in HCC liver transplant candidates**](#64-french-team-proposes-explainable-machine-learning-framework-for-accurate-mortality-prediction-in-hcc-liver-transplant-candidates)
  - [**65. Stanford University proposes Merlin, the first native 3D abdominal CT vision-language model**](#65-stanford-university-proposes-merlin-the-first-native-3d-abdominal-ct-vision-language-model)
- [**AI+ Materials Chemistry**](#ai-materials-chemistry)
  - [**1. High-throughput computational framework generates 120,000 novel MOF candidates in 33 minutes**](#1-high-throughput-computational-framework-generates-120000-novel-mof-candidates-in-33-minutes)
  - [**2. Machine learning algorithm screens P-SOC electrode materials**](#2-machine-learning-algorithm-screens-p-soc-electrode-materials)
  - [**3. SEN machine learning model achieves high-accuracy material property predictions**](#3-sen-machine-learning-model-achieves-high-accuracy-material-property-predictions)
  - [**4. Deep learning tool GNoME discovers 2.2 million new crystals**](#4-deep-learning-tool-gnome-discovers-22-million-new-crystals)
  - [**5. Field-induced recursively embedded atom neural network accurately describes external field strength and direction changes**](#5-field-induced-recursively-embedded-atom-neural-network-accurately-describes-external-field-strength-and-direction-changes)
  - [**6. Machine learning predicts water adsorption isotherms of porous materials**](#6-machine-learning-predicts-water-adsorption-isotherms-of-porous-materials)
  - [**7. Using machine learning to optimize co-catalysts for BiVO(4) photoanodes**](#7-using-machine-learning-to-optimize-co-catalysts-for-bivo4-photoanodes)
  - [**8. RetroExplainer algorithm performs retrosynthesis prediction based on deep learning**](#8-retroexplainer-algorithm-performs-retrosynthesis-prediction-based-on-deep-learning)
  - [**9. Deep neural networks + NLP used to develop corrosion-resistant alloys**](#9-deep-neural-networks--nlp-used-to-develop-corrosion-resistant-alloys)
  - [**10. Deep learning determines materials' internal structures through surface observations**](#10-deep-learning-determines-materials-internal-structures-through-surface-observations)
  - [**11. Developing 3 new materials using innovative X-ray scintillators**](#11-developing-3-new-materials-using-innovative-x-ray-scintillators)
  - [**12. Semi-supervised learning extracts hidden information from unlabeled data**](#12-semi-supervised-learning-extracts-hidden-information-from-unlabeled-data)
  - [**13. Automated knowledge extraction based on AutoML**](#13-automated-knowledge-extraction-based-on-automl)
  - [**14. Uni-MOF: A machine learning model predicting adsorption behavior in 3D MOF materials**](#14-uni-mof-a-machine-learning-model-predicting-adsorption-behavior-in-3d-mof-materials)
  - [**15. Microelectronics accelerates towards the post-Moore era! Integrating DNN with nanomembrane technology to precisely analyze incident light angles**](#15-microelectronics-accelerates-towards-the-post-moore-era-integrating-dnn-with-nanomembrane-technology-to-precisely-analyze-incident-light-angles)
  - [**16. Reshaping lithium battery performance boundaries, proposing a simplified electrochemical model based on ensemble learning**](#16-reshaping-lithium-battery-performance-boundaries-proposing-a-simplified-electrochemical-model-based-on-ensemble-learning)
  - [**17. The strongest iron-based superconducting magnet born via machine learning**](#17-the-strongest-iron-based-superconducting-magnet-born-via-machine-learning)
  - [**18. Neural networks replace Density Functional Theory! Universal materials model achieves ultra-precise predictions**](#18-neural-networks-replace-density-functional-theory-universal-materials-model-achieves-ultra-precise-predictions)
  - [**19. Neural network density functional framework opens the black box of matter's electronic structure prediction**](#19-neural-network-density-functional-framework-opens-the-black-box-of-matters-electronic-structure-prediction)
  - [**20. First fully forward mode training architecture for optical computing using neural networks achieves major breakthrough in domestic optical chips**](#20-first-fully-forward-mode-training-architecture-for-optical-computing-using-neural-networks-achieves-major-breakthrough-in-domestic-optical-chips)
  - [**21. Chemistry LLM ChemLLM covers 7 million QA data, professional capabilities rival GPT-4**](#21-chemistry-llm-chemllm-covers-7-million-qa-data-professional-capabilities-rival-gpt-4)
  - [**22. Wafer-scale producible AI-adaptive micro-spectrometers**](#22-wafer-scale-producible-ai-adaptive-micro-spectrometers)
  - [**23. GNNOpt model identifies hundreds of solar cell and quantum material candidates**](#23-gnnopt-model-identifies-hundreds-of-solar-cell-and-quantum-material-candidates)
  - [**24. Open OMat24 dataset contains 110 million DFT calculation results**](#24-open-omat24-dataset-contains-110-million-dft-calculation-results)
  - [**25. Novel refractory high-entropy alloy synthesized via machine learning boasts excellent room-temperature ductility**](#25-novel-refractory-high-entropy-alloy-synthesized-via-machine-learning-boasts-excellent-room-temperature-ductility)
  - [**26. Material generative model FlowLLM features a dataset covering over 45k materials**](#26-material-generative-model-flowllm-features-a-dataset-covering-over-45k-materials)
  - [**27. Using active learning to identify 14,000 high-entropy oxides, successfully screening 4 high-activity hydrogen evolution catalysts**](#27-using-active-learning-to-identify-14000-high-entropy-oxides-successfully-screening-4-high-activity-hydrogen-evolution-catalysts)
  - [**28. Deep learning model BETE-NET boosts superconducting material search efficiency by 5x**](#28-deep-learning-model-bete-net-boosts-superconducting-material-search-efficiency-by-5x)
  - [**29. Gradient Boosting Decision Tree (GBDT) technology further improves high-precision prediction of high-entropy alloy oxidation resistance**](#29-gradient-boosting-decision-tree-gbdt-technology-further-improves-high-precision-prediction-of-high-entropy-alloy-oxidation-resistance)
  - [**30. Molecular design framework RingFormer more precisely predicts organic material molecular optoelectronic properties**](#30-molecular-design-framework-ringformer-more-precisely-predicts-organic-material-molecular-optoelectronic-properties)
  - [**31. Inorganic retrosynthesis planning method Retrieval-Retro improves inorganic material synthesis efficiency and accuracy**](#31-inorganic-retrosynthesis-planning-method-retrieval-retro-improves-inorganic-material-synthesis-efficiency-and-accuracy)
  - [**32. Using large models to decipher hydride solid-state electrolyte conduction mechanisms, establishing a reliable activation energy prediction model**](#32-using-large-models-to-decipher-hydride-solid-state-electrolyte-conduction-mechanisms-establishing-a-reliable-activation-energy-prediction-model)
  - [**33. Tera-scale mass spectrometry data search enabled by machine learning uncovers unknown chemical reactions**](#33-tera-scale-mass-spectrometry-data-search-enabled-by-machine-learning-uncovers-unknown-chemical-reactions)
  - [**34. Generative AI structure solution method PXRDnet based on diffusion models successfully solves 200 complex simulated nanocrystals**](#34-generative-ai-structure-solution-method-pxrdnet-based-on-diffusion-models-successfully-solves-200-complex-simulated-nanocrystals)
  - [**35. DreaMS model covers 200 million molecular mass spectra, building the world's largest mass spec dataset GeMS**](#35-dreams-model-covers-200-million-molecular-mass-spectra-building-the-worlds-largest-mass-spec-dataset-gems)
  - [**36. Equivariant machine learning framework accelerates large-scale electric field simulations of materials**](#36-equivariant-machine-learning-framework-accelerates-large-scale-electric-field-simulations-of-materials)
  - [**37. Multi-source data integration method screens 25 types of cement clinker alternatives, equivalent to reducing 1.2 billion tons of greenhouse gases**](#37-multi-source-data-integration-method-screens-25-types-of-cement-clinker-alternatives-equivalent-to-reducing-12-billion-tons-of-greenhouse-gases)
  - [**38. UNIMATE achieves unified modeling of topology generation/property prediction for the first time**](#38-unimate-achieves-unified-modeling-of-topology-generationproperty-prediction-for-the-first-time)
  - [**39. All-atom diffusion Transformer framework enables unified generation of periodic and aperiodic atomic systems for the first time**](#39-all-atom-diffusion-transformer-framework-enables-unified-generation-of-periodic-and-aperiodic-atomic-systems-for-the-first-time)
  - [**40. FASTSOLV model realizes small molecule solubility prediction at any temperature, accelerating inference speed by 50x**](#40-fastsolv-model-realizes-small-molecule-solubility-prediction-at-any-temperature-accelerating-inference-speed-by-50x)
  - [**41. Novel method based on multimodal machine learning models predicts material properties without complete crystal structures**](#41-novel-method-based-on-multimodal-machine-learning-models-predicts-material-properties-without-complete-crystal-structures)
  - [**42. AI model CGformer innovatively integrates global attention mechanisms, aiding high-entropy material R&D**](#42-ai-model-cgformer-innovatively-integrates-global-attention-mechanisms-aiding-high-entropy-material-rd)
  - [**43. Novel structural constraint integration method SCIGEN adapts to any pre-trained diffusion model**](#43-novel-structural-constraint-integration-method-scigen-adapts-to-any-pre-trained-diffusion-model)
  - [**44. Physically-informed generative AI model SpectroGen requires only single modality input to achieve cross-modal generation with 99% experimental correlation**](#44-physically-informed-generative-ai-model-spectrogen-requires-only-single-modality-input-to-achieve-cross-modal-generation-with-99-experimental-correlation)
  - [**45. MOF-ChemUnity reconstructs MOF panoramic knowledge, pushing material discovery into the "Explainable AI" era**](#45-mof-chemunity-reconstructs-mof-panoramic-knowledge-pushing-material-discovery-into-the-explainable-ai-era)
  - [**46. Lightweight universal potential model PET-MAD released, achieving dedicated model-level precision with minimal samples**](#46-lightweight-universal-potential-model-pet-mad-released-achieving-dedicated-model-level-precision-with-minimal-samples)
  - [**47. AI system ChemOntology released, halving reaction path search costs by integrating chemical knowledge**](#47-ai-system-chemontology-released-halving-reaction-path-search-costs-by-integrating-chemical-knowledge)
  - [**48. Princeton and others jointly propose LLM method for predicting MOF free energy, highly accurately assessing synthesis feasibility**](#48-princeton-and-others-jointly-propose-llm-method-for-predicting-mof-free-energy-highly-accurately-assessing-synthesis-feasibility)
  - [**49. Yale University team proposes MOSAIC model, coordinating LLMs to generate highly reliable chemical synthesis schemes**](#49-yale-university-team-proposes-mosaic-model-coordinating-llms-to-generate-highly-reliable-chemical-synthesis-schemes)
  - [**50. MIT and others propose diffusion model DiffSyn, enabling generative planning of material synthesis pathways**](#50-mit-and-others-propose-diffusion-model-diffsyn-enabling-generative-planning-of-material-synthesis-pathways)
  - [**51. University of Michigan and Farasis Energy jointly propose "Discovery Learning" method, drastically shortening battery life prediction cycles**](#51-university-of-michigan-and-farasis-energy-jointly-propose-discovery-learning-method-drastically-shortening-battery-life-prediction-cycles)
  - [**52. Cornell University proposes SCAN framework, highly accurately predicting and explaining battery electrolyte performance**](#52-cornell-university-proposes-scan-framework-highly-accurately-predicting-and-explaining-battery-electrolyte-performance)
  - [**53. MIT proposes foundation large model DefectNet for non-destructive characterization and quantification of internal material defects**](#53-mit-proposes-foundation-large-model-defectnet-for-non-destructive-characterization-and-quantification-of-internal-material-defects)
  - [**54. Cornell University proposes multi-agent platform EMSeek, achieving full-pipeline automated analysis of electron microscopy images**](#54-cornell-university-proposes-multi-agent-platform-emseek-achieving-full-pipeline-automated-analysis-of-electron-microscopy-images)
- [**AI+ Zoology-Botany**](#ai-zoology-botany)
  - [**1. SBeA analyzes animal social behaviors based on a few-shot learning framework**](#1-sbea-analyzes-animal-social-behaviors-based-on-a-few-shot-learning-framework)
  - [**2. Deep learning method based on Siamese networks automatically captures embryonic development processes**](#2-deep-learning-method-based-on-siamese-networks-automatically-captures-embryonic-development-processes)
  - [**3. Systematic pipeline for collecting plant phenotype data via drones to predict optimal harvest dates**](#3-systematic-pipeline-for-collecting-plant-phenotype-data-via-drones-to-predict-optimal-harvest-dates)
  - [**4. AI camera alert system accurately distinguishes tigers from other species**](#4-ai-camera-alert-system-accurately-distinguishes-tigers-from-other-species)
  - [**5. Using Labrador retriever data and comparing 3 models reveals behavioral traits affecting detection dogs' performance**](#5-using-labrador-retriever-data-and-comparing-3-models-reveals-behavioral-traits-affecting-detection-dogs-performance)
  - [**6. Multi-species image recognition model based on ArcFace Classification Head for face recognition**](#6-multi-species-image-recognition-model-based-on-arcface-classification-head-for-face-recognition)
  - [**7. Monitoring cherry blossom blooming in Japan using Python API and computer vision API**](#7-monitoring-cherry-blossom-blooming-in-japan-using-python-api-and-computer-vision-api)
  - [**8. Machine learning-based population genetics method reveals the formation mechanism of grape flavors**](#8-machine-learning-based-population-genetics-method-reveals-the-formation-mechanism-of-grape-flavors)
  - [**9. Review: Unlocking bioinformatics research more efficiently with AI**](#9-review-unlocking-bioinformatics-research-more-efficiently-with-ai)
  - [**10. BirdFlow model accurately predicts flight paths of migratory birds**](#10-birdflow-model-accurately-predicts-flight-paths-of-migratory-birds)
  - [**11. New whale bioacoustics model identifies 8 cetacean species**](#11-new-whale-bioacoustics-model-identifies-8-cetacean-species)
  - [**12. Machine learning isolates the sperm whale phonetic alphabet, highly similar to human language with stronger information-carrying capacity**](#12-machine-learning-isolates-the-sperm-whale-phonetic-alphabet-highly-similar-to-human-language-with-stronger-information-carrying-capacity)
  - [**13. PlantLncBoost model achieves up to 96% accuracy in cross-species lncRNA prediction**](#13-plantlncboost-model-achieves-up-to-96-accuracy-in-cross-species-lncrna-prediction)
  - [**14. Perch 2.0 covers nearly 15,000 species, refreshing SOTA in bioacoustic classification detection**](#14-perch-20-covers-nearly-15000-species-refreshing-sota-in-bioacoustic-classification-detection)
- [**AI+ Agriculture-Forestry-Animal husbandry**](#ai-agriculture-forestry-animal-husbandry)
  - [**1. Using Convolutional Neural Networks for rapid and accurate rice yield estimation**](#1-using-convolutional-neural-networks-for-rapid-and-accurate-rice-yield-estimation)
  - [**2. Model designed via YOLOv5 algorithm monitors sow posture and piglet birth**](#2-model-designed-via-yolov5-algorithm-monitors-sow-posture-and-piglet-birth)
  - [**3. Combining laboratory observation and machine learning to prove that ultrasonic sounds emitted by stressed tomato and tobacco plants can travel in air**](#3-combining-laboratory-observation-and-machine-learning-to-prove-that-ultrasonic-sounds-emitted-by-stressed-tomato-and-tobacco-plants-can-travel-in-air)
  - [**4. Drone + AI image analysis detects forestry pests**](#4-drone--ai-image-analysis-detects-forestry-pests)
  - [**5. Computer vision + deep learning developed for a dairy cow lameness detection system**](#5-computer-vision--deep-learning-developed-for-a-dairy-cow-lameness-detection-system)
- [**AI+ Meteorology**](#ai-meteorology)
  - [**1. Review: Data-driven machine learning weather forecasting models**](#1-review-data-driven-machine-learning-weather-forecasting-models)
  - [**2. Review: Collecting data from hailstorm centers and predicting extreme weather using large models**](#2-review-collecting-data-from-hailstorm-centers-and-predicting-extreme-weather-using-large-models)
  - [**3. Creating new algorithms to accurately predict extreme precipitation using global storm-resolving simulations and machine learning**](#3-creating-new-algorithms-to-accurately-predict-extreme-precipitation-using-global-storm-resolving-simulations-and-machine-learning)
  - [**4. Random Forest-based machine learning model CSU-MLP predicts medium-range severe weather**](#4-random-forest-based-machine-learning-model-csu-mlp-predicts-medium-range-severe-weather)
  - [**5. End-to-end data-driven weather forecasting system Aardvark Weather speeds up predictions by dozens of times compared to traditional methods**](#5-end-to-end-data-driven-weather-forecasting-system-aardvark-weather-speeds-up-predictions-by-dozens-of-times-compared-to-traditional-methods)
  - [**6. Machine learning weather forecasting system FCN3 supports ultra-fast single-GPU inference**](#6-machine-learning-weather-forecasting-system-fcn3-supports-ultra-fast-single-gpu-inference)
  - [**7. Indian monsoon forecasting model based on 36 weather stations achieves city-scale fine forecasting**](#7-indian-monsoon-forecasting-model-based-on-36-weather-stations-achieves-city-scale-fine-forecasting)
  - [**8. ACE2 completes a 4-month seasonal forecast in just 2 minutes**](#8-ace2-completes-a-4-month-seasonal-forecast-in-just-2-minutes)
  - [**9. Incremental weather forecasting model VA-MoE released, achieving SOTA performance with 75% parameter reduction**](#9-incremental-weather-forecasting-model-va-moe-released-achieving-sota-performance-with-75-parameter-reduction)
  - [**10. Elucidated Rolling Diffusion Model (ERDM) released, solving long-term forecasting challenges and maintaining a lead over EDM baselines in medium-to-long term forecasts**](#10-elucidated-rolling-diffusion-model-erdm-released-solving-long-term-forecasting-challenges-and-maintaining-a-lead-over-edm-baselines-in-medium-to-long-term-forecasts)
  - [**11. Novel latent diffusion model OmniCast released, resolving error accumulation in autoregressive weather forecasting models**](#11-novel-latent-diffusion-model-omnicast-released-resolving-error-accumulation-in-autoregressive-weather-forecasting-models)
  - [**12. NVIDIA proposes a novel long-range distillation method, breaking AI bottlenecks in long-term weather forecasting**](#12-nvidia-proposes-a-novel-long-range-distillation-method-breaking-ai-bottlenecks-in-long-term-weather-forecasting)
  - [**13. Joint team proposes Graph Neural Network model SeaCast, achieving ultra-fast regional ocean forecasting**](#13-joint-team-proposes-graph-neural-network-model-seacast-achieving-ultra-fast-regional-ocean-forecasting)
- [**AI+ Astronomy**](#ai-astronomy)
  - [**1. PRIMO algorithm learns light propagation rules around black holes to reconstruct sharper black hole images**](#1-primo-algorithm-learns-light-propagation-rules-around-black-holes-to-reconstruct-sharper-black-hole-images)
  - [**2. Training computer vision algorithms with simulated data to sharpen and "restore" astronomical images**](#2-training-computer-vision-algorithms-with-simulated-data-to-sharpen-and-restore-astronomical-images)
  - [**3. Using unsupervised machine learning algorithm Astronomaly to find previously overlooked anomalies**](#3-using-unsupervised-machine-learning-algorithm-astronomaly-to-find-previously-overlooked-anomalies)
  - [**4. Machine learning-based method for CME identification and parameter extraction**](#4-machine-learning-based-method-for-cme-identification-and-parameter-extraction)
  - [**5. Deep learning discovers 107 cases of neutral carbon absorption lines**](#5-deep-learning-discovers-107-cases-of-neutral-carbon-absorption-lines)
  - [**6. StarFusion model achieves high spatial resolution image prediction**](#6-starfusion-model-achieves-high-spatial-resolution-image-prediction)
  - [**7. Satellite image generation method developed based on SD3, constructing the largest remote sensing dataset to date, EcoMapper**](#7-satellite-image-generation-method-developed-based-on-sd3-constructing-the-largest-remote-sensing-dataset-to-date-ecomapper)
  - [**8. Geospatial AI Earth AI focuses on 3 core data types, improving geospatial reasoning capabilities by 64%**](#8-geospatial-ai-earth-ai-focuses-on-3-core-data-types-improving-geospatial-reasoning-capabilities-by-64)
  - [**9. The first astronomical multimodal foundation model AION-1 is born, pre-trained on 200 million astronomical targets**](#9-the-first-astronomical-multimodal-foundation-model-aion-1-is-born-pre-trained-on-200-million-astronomical-targets)
  - [**10. Novel data-driven pipeline precisely identifies 7 rare lensed samples from 810,000 quasars using CNN**](#10-novel-data-driven-pipeline-precisely-identifies-7-rare-lensed-samples-from-810000-quasars-using-cnn)
  - [**11. ESA team proposes semi-supervised method AnomalyMatch to efficiently screen rare celestial bodies from nearly 100 million Hubble records**](#11-esa-team-proposes-semi-supervised-method-anomalymatch-to-efficiently-screen-rare-celestial-bodies-from-nearly-100-million-hubble-records)
  - [**12. University of Warwick proposes the RAVEN validation pipeline, confirming 118 new exoplanets**](#12-university-of-warwick-proposes-the-raven-validation-pipeline-confirming-118-new-exoplanets)
  - [**13. University of Warwick proposes an ensemble learning framework to highly accurately predict asteroseismic parameters for δ Scuti stars**](#13-university-of-warwick-proposes-an-ensemble-learning-framework-to-highly-accurately-predict-asteroseismic-parameters-for-δ-scuti-stars)
  - [**14. Spanish research team proposes the StreakMind system, utilizing AI to automatically detect satellite streaks in astronomical images**](#14-spanish-research-team-proposes-the-streakmind-system-utilizing-ai-to-automatically-detect-satellite-streaks-in-astronomical-images)
- [**AI+ Natural Disaster**](#ai-natural-disaster)
  - [**1. Machine learning predicts land subsidence risk over the next 40 years**](#1-machine-learning-predicts-land-subsidence-risk-over-the-next-40-years)
  - [**2. Semantic segmentation model SCDUNet++ used for landslide mapping**](#2-semantic-segmentation-model-scdunet-used-for-landslide-mapping)
  - [**3. Neural networks convert 2D solar images into 3D reconstructed images**](#3-neural-networks-convert-2d-solar-images-into-3d-reconstructed-images)
  - [**4. Additive neural networks analyze influencing factors in natural disasters**](#4-additive-neural-networks-analyze-influencing-factors-in-natural-disasters)
  - [**5. Using Explainable AI to analyze various geographical factors in Gippsland, Australia**](#5-using-explainable-ai-to-analyze-various-geographical-factors-in-gippsland-australia)
  - [**6. Machine learning-based flood forecasting model**](#6-machine-learning-based-flood-forecasting-model)
  - [**7. ED-DLSTM achieves flood prediction in unmonitored areas**](#7-ed-dlstm-achieves-flood-prediction-in-unmonitored-areas)
  - [**8. ChloroFormer model provides early warning of marine algal blooms**](#8-chloroformer-model-provides-early-warning-of-marine-algal-blooms)
  - [**9. The first marine large language model OceanGPT accepted by ACL 2024! Underwater embodied AI becomes reality**](#9-the-first-marine-large-language-model-oceangpt-accepted-by-acl-2024-underwater-embodied-ai-becomes-reality)
  - [**10. AI predicts global warming trends**](#10-ai-predicts-global-warming-trends)
  - [**11. New GeoAI model explains surface heat flow distribution on the Tibetan Plateau**](#11-new-geoai-model-explains-surface-heat-flow-distribution-on-the-tibetan-plateau)
  - [**12. "WenHai" marine environment intelligent forecasting large model outperforms numerical marine forecasting**](#12-wenhai-marine-environment-intelligent-forecasting-large-model-outperforms-numerical-marine-forecasting)
  - [**13. University of Minnesota proposes knowledge-guided machine learning model FHNN, realizing high-precision flood forecasting**](#13-university-of-minnesota-proposes-knowledge-guided-machine-learning-model-fhnn-realizing-high-precision-flood-forecasting)
  - [**14. Google releases version 2 of its global flood forecasting system, significantly extending valid forecast times**](#14-google-releases-version-2-of-its-global-flood-forecasting-system-significantly-extending-valid-forecast-times)
- [**Others**](#others)
  - [**1. TacticAI football assistant hits 90% practical utility in tactical layouts**](#1-tacticai-football-assistant-hits-90-practical-utility-in-tactical-layouts)
  - [**2. Denoising diffusion model SPDiff enables long-range crowd movement simulation**](#2-denoising-diffusion-model-spdiff-enables-long-range-crowd-movement-simulation)
  - [**3. Intelligent scientific facilities drive paradigm shifts in research**](#3-intelligent-scientific-facilities-drive-paradigm-shifts-in-research)
  - [**4. DeepSymNet represents symbolic expressions based on supervised learning**](#4-deepsymnet-represents-symbolic-expressions-based-on-supervised-learning)
  - [**5. Large language model ChipNeMo assists engineers in chip design**](#5-large-language-model-chipnemo-assists-engineers-in-chip-design)
  - [**6. AlphaGeometry can solve geometry problems**](#6-alphageometry-can-solve-geometry-problems)
  - [**7. Reinforcement learning applied to urban spatial planning**](#7-reinforcement-learning-applied-to-urban-spatial-planning)
  - [**8. ChatArena framework: Playing Werewolf with Large Language Models**](#8-chatarena-framework-playing-werewolf-with-large-language-models)
  - [**9. Review: 30 scholars co-publish in Nature, 10-year retrospective deconstructs how AI reshapes scientific paradigms**](#9-review-30-scholars-co-publish-in-nature-10-year-retrospective-deconstructs-how-ai-reshapes-scientific-paradigms)
  - [**10. Ithaca assists epigraphers in text restoration, chronological attribution, and geographical attribution**](#10-ithaca-assists-epigraphers-in-text-restoration-chronological-attribution-and-geographical-attribution)
  - [**11. AI in forward and inverse problems of meta-optics, data analysis based on metasurface systems**](#11-ai-in-forward-and-inverse-problems-of-meta-optics-data-analysis-based-on-metasurface-systems)
  - [**12. A new geospatial artificial intelligence method: Geographically Neural Network Weighted Logistic Regression**](#12-a-new-geospatial-artificial-intelligence-method-geographically-neural-network-weighted-logistic-regression)
  - [**13. Using diffusion models to generate neural network parameters, transforming spatiotemporal few-shot learning into a diffusion model pre-training problem**](#13-using-diffusion-models-to-generate-neural-network-parameters-transforming-spatiotemporal-few-shot-learning-into-a-diffusion-model-pre-training-problem)
  - [**14. Latest AI4S insights from Fei-Fei Li's team: 16 innovative technologies summarized, covering biology/materials/healthcare/diagnostics**](#14-latest-ai4s-insights-from-fei-fei-lis-team-16-innovative-technologies-summarized-covering-biologymaterialshealthcarediagnostics)
  - [**15. Accurate prediction of Wuhan housing prices! osp-GNNWR model accurately describes complex spatial processes and geographical phenomena**](#15-accurate-prediction-of-wuhan-housing-prices-osp-gnnwr-model-accurately-describes-complex-spatial-processes-and-geographical-phenomena)
  - [**16. Introducing zero-shot learning to release a conditional diffusion model optimized for oracle bone script decipherment**](#16-introducing-zero-shot-learning-to-release-a-conditional-diffusion-model-optimized-for-oracle-bone-script-decipherment)
  - [**17. Stanford/Apple and 23 other institutions release the DCLM benchmark; foundation model performs on par with Llama3 8B**](#17-stanfordapple-and-23-other-institutions-release-the-dclm-benchmark-foundation-model-performs-on-par-with-llama3-8b)
  - [**18. PoCo solves the data source heterogeneity dilemma, enabling robots to execute multi-tasks flexibly**](#18-poco-solves-the-data-source-heterogeneity-dilemma-enabling-robots-to-execute-multi-tasks-flexibly)
  - [**19. Containing 140,000 images! Oracle bone script dataset helps team win ACL Best Paper Award**](#19-containing-140000-images-oracle-bone-script-dataset-helps-team-win-acl-best-paper-award)
  - [**20. Proposing a channel prediction scheme based on pre-trained LLMs, GPT-2 empowers the physical layer of wireless communications**](#20-proposing-a-channel-prediction-scheme-based-on-pre-trained-llms-gpt-2-empowers-the-physical-layer-of-wireless-communications)
  - [**21. The first Generative Adversarial Network model for multi-stitch embroidery**](#21-the-first-generative-adversarial-network-model-for-multi-stitch-embroidery)
  - [**22. Fast Automated Scanning Toolkit (FAST) efficiently acquires sample information**](#22-fast-automated-scanning-toolkit-fast-efficiently-acquires-sample-information)
  - [**23. Population Dynamics Foundation Model PDFM open-sourced, precisely predicting US unemployment and poverty rates**](#23-population-dynamics-foundation-model-pdfm-open-sourced-precisely-predicting-us-unemployment-and-poverty-rates)
  - [**24. Deep learning model CatGWR estimates spatial non-stationarity**](#24-deep-learning-model-catgwr-estimates-spatial-non-stationarity)
  - [**25. World's first VR exercise intervention system REVERIE reshapes youth brain-body-mind health**](#25-worlds-first-vr-exercise-intervention-system-reverie-reshapes-youth-brain-body-mind-health)
  - [**26. Based on over 176k inscription data, Aeneas achieves arbitrary-length restoration of ancient Roman inscriptions for the first time**](#26-based-on-over-176k-inscription-data-aeneas-achieves-arbitrary-length-restoration-of-ancient-roman-inscriptions-for-the-first-time)
  - [**27. Panoramic video generation framework PanoWan also handles zero-shot video editing**](#27-panoramic-video-generation-framework-panowan-also-handles-zero-shot-video-editing)
  - [**28. YOLOv11-based ceramic classification intelligent framework integrates visual modeling and economic analysis, achieving artifact classification and value estimation**](#28-yolov11-based-ceramic-classification-intelligent-framework-integrates-visual-modeling-and-economic-analysis-achieving-artifact-classification-and-value-estimation)
  - [**29. "Microwave Brain" chip born, simultaneously processing ultra-high-speed data and wireless signals with 75% accuracy at 176 milliwatts power**](#29-microwave-brain-chip-born-simultaneously-processing-ultra-high-speed-data-and-wireless-signals-with-75-accuracy-at-176-milliwatts-power)
  - [**30. Spatiotemporal imputation and prediction model STIMP released, realizing precise predictions of coastal Chlorophyll-a distribution**](#30-spatiotemporal-imputation-and-prediction-model-stimp-released-realizing-precise-predictions-of-coastal-chlorophyll-a-distribution)
  - [**31. MIT and others achieve high-precision prediction of plasma dynamics under few-shot conditions based on machine learning**](#31-mit-and-others-achieve-high-precision-prediction-of-plasma-dynamics-under-few-shot-conditions-based-on-machine-learning)
  - [**32. Reac-Discovery fuses mathematical modeling, machine learning, and automated experiments to solve the universality challenge of self-driving laboratory systems**](#32-reac-discovery-fuses-mathematical-modeling-machine-learning-and-automated-experiments-to-solve-the-universality-challenge-of-self-driving-laboratory-systems)
  - [**33. The first neuron modeling framework NOBLE validated by human cortical data is introduced**](#33-the-first-neuron-modeling-framework-noble-validated-by-human-cortical-data-is-introduced)
  - [**34. Image geolocation framework LocDiff goes online, enabling grid-free and reference-library-free global precision positioning**](#34-image-geolocation-framework-locdiff-goes-online-enabling-grid-free-and-reference-library-free-global-precision-positioning)
  - [**35. Machine learning combined with py-GC-MS precisely identifies evidence of life in Archean rocks**](#35-machine-learning-combined-with-py-gc-ms-precisely-identifies-evidence-of-life-in-archean-rocks)
  - [**36. Tsinghua University team proposes neuro-symbolic regression method ND² to automatically derive complex network dynamics formulas**](#36-tsinghua-university-team-proposes-neuro-symbolic-regression-method-nd-to-automatically-derive-complex-network-dynamics-formulas)
  - [**37. Zhejiang University team proposes geologically constrained mineral prospectivity prediction method, explicitly depicting mineralization anisotropy**](#37-zhejiang-university-team-proposes-geologically-constrained-mineral-prospectivity-prediction-method-explicitly-depicting-mineralization-anisotropy)
  - [**38. Tsinghua and UChicago team publishes in Nature: AI tools expand scientists' impact but contract science's focus**](#38-tsinghua-and-uchicago-team-publishes-in-nature-ai-tools-expand-scientists-impact-but-contract-sciences-focus)
  - [**39. UC team proposes AI-augmented chip-scale spectrometer, achieving high spectral fidelity in an ultra-small volume**](#39-uc-team-proposes-ai-augmented-chip-scale-spectrometer-achieving-high-spectral-fidelity-in-an-ultra-small-volume)
  - [**40. US DOE Oak Ridge National Lab proposes D-CHAG method, significantly reducing memory footprint for multi-channel foundation models**](#40-us-doe-oak-ridge-national-lab-proposes-d-chag-method-significantly-reducing-memory-footprint-for-multi-channel-foundation-models)
  - [**41. Polymathic AI team proposes continuum foundation model Walrus, setting records in cross-domain simulation performance**](#41-polymathic-ai-team-proposes-continuum-foundation-model-walrus-setting-records-in-cross-domain-simulation-performance)
  - [**42. EPFL proposes novel architecture DYNAMI-CAL GraphNet, a physics-informed GNN accurately modeling multi-body dynamics**](#42-epfl-proposes-novel-architecture-dynami-cal-graphnet-a-physics-informed-gnn-accurately-modeling-multi-body-dynamics)
  - [**43. MIT proposes novel method Wave-Former, achieving high-precision 3D reconstruction of completely occluded objects**](#43-mit-proposes-novel-method-wave-former-achieving-high-precision-3d-reconstruction-of-completely-occluded-objects)
  - [**44. MIT proposes DRiffusion draft-and-refine parallel framework, realizing lossless acceleration for diffusion model inference**](#44-mit-proposes-driffusion-draft-and-refine-parallel-framework-realizing-lossless-acceleration-for-diffusion-model-inference)
  - [**45. Technion - Israel Institute of Technology proposes Task Tokens, allowing behavior foundation models to flexibly adapt to specific tasks**](#45-technion---israel-institute-of-technology-proposes-task-tokens-allowing-behavior-foundation-models-to-flexibly-adapt-to-specific-tasks)
  - [**46. MIT and others propose EnergAIzer framework, achieving fast and accurate GPU power estimation for AI workloads**](#46-mit-and-others-propose-energaizer-framework-achieving-fast-and-accurate-gpu-power-estimation-for-ai-workloads)
  - [**47. UIUC proposes heterogeneous agent framework Eywa, breaking through the limits of language-centric large models**](#47-uiuc-proposes-heterogeneous-agent-framework-eywa-breaking-through-the-limits-of-language-centric-large-models)
  - [**48. Stanford University and others use LSTM surrogate models to achieve 252x accelerated simulation of second-order nonlinear optics**](#48-stanford-university-and-others-use-lstm-surrogate-models-to-achieve-252x-accelerated-simulation-of-second-order-nonlinear-optics)

## **Foreword**

Since 2020, scientific projects represented by AlphaFold have pushed AI for Science (AI4S) to the main stage of AI applications. In recent years, from biopharmaceuticals to astronomy and meteorology, and then to fundamental disciplines like materials chemistry, all have become new battlegrounds for AI.

As an increasing number of interdisciplinary talents begin to apply technologies such as machine learning and deep learning to data processing and model building in their research fields, coupled with the strengthening collaboration of cross-disciplinary research teams, the capabilities of AI4S are being noticed by more scientific researchers. However, it has not yet achieved the goal of large-scale application. Many issues urgently need to be resolved, such as improving the reproducibility of related research, lowering the technical threshold, and improving data quality.

Currently, in addition to universities and research institutions actively exploring AI4S, many governments and leading technology companies have also noticed the potential of AI to revolutionize scientific research and have initiated relevant policy guidance and layouts. It can be said that AI4S is the undeniable general trend.

As one of the earliest communities to pay attention to AI for Science, "HyperAI" is happy to share the latest research progress and results universally while accompanying the industry's growth. We hope that by interpreting cutting-edge papers and policies, more teams can see the help AI brings to scientific research, contributing to the development of AI for Science.

To date, HyperAI has interpreted and shared nearly 200 papers. For ease of retrieval, we have classified the articles by discipline, displayed the publishing journals and dates, and extracted keywords (research teams, related research, datasets, etc.). You can click the titles to jump to the Research highlight page of the paper (which contains the full paper download link).

This document will be presented as an open-source project. We will continuously update the interpretation articles, and we also welcome everyone to submit excellent research results. If your team/research group has reporting needs, you can add WeChat: 神经星星 (WeChat ID: Hyperai01).

## **AI+ Biopharmaceutical**

### **1. [AdaDR outperforms multiple benchmark methods in drug repositioning](https://hyper.ai/news/30434)**

- **Research highlight:** [https://hyper.ai/news/30434](https://hyper.ai/news/30434)
- **Research Team:** Min Li's Research Team at Central South University
- **Related Research:** Gdataset, Cdataset, Ldataset, LRSSL dataset, GCNs framework, AdaDR
- **Published Journal:** Bioinformatics, 2024.01
- **Paper Link:** [Drug repositioning with adaptive graph convolutional networks](https://academic.oup.com/bioinformatics/article/40/1/btad748/7467059)

### **2. [IMN4NPD accelerates the dereplication of extensive clusters in molecular networks, providing annotations for self-loops and paired nodes](https://hyper.ai/news/30363)**

- **Research highlight:** [https://hyper.ai/news/30363](https://hyper.ai/news/30363)
- **Research Team:** Shao Liu's Research Team at Central South University
- **Related Research:** MS/MS spectral database, Structure database, molDiscovery, NPClassifier, t-SNE
- **Published Journal:** Analytical Chemistry, 2024.02
- **Paper Link:** [IMN4NPD: An Integrated Molecular Networking Workflow for Natural Product Dereplication](https://doi.org/10.1021/acs.analchem.3c04746)

### **3. [Deep generative model MIDAS for mosaic integration of single-cell multi-omics data](https://hyper.ai/news/29785)**

- **Research highlight:** [https://hyper.ai/news/29785](https://hyper.ai/news/29785)
- **Research Team:** Xiaomin Ying's Research Team at the Academy of Military Medical Sciences
- **Related Research:** IPBMC dataset, dogma-full dataset, teadog-full dataset, MMIDAS, self-supervised learning, information-theoretic approaches, deep neural networks, SGVB, single-cell multi-omics mosaic data
- **Published Journal:** Nature Biotechnology, 2024.01
- **Paper Link:** [Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS](https://www.nature.com/articles/s41587-023-02040-y)

### **4. [ResGen: A 3D molecular generation model based on protein pockets](https://hyper.ai/news/29026)**

- **Research highlight:** [https://hyper.ai/news/29026](https://hyper.ai/news/29026)
- **Research Team:** Tingjun Hou's Research Team at Zhejiang University
- **Related Research:** CrossDock2020 dataset, global autoregressive, atom autoregressive, parallel multiscale modeling, SBMG. 8 times faster than state-of-the-art techniques.
- **Published Journal:** Nature Machine Intelligence, 2023.09
- **Paper Link:** [ResGen is a pocket-aware 3D molecular generation model based on parallel multiscale modelling](https://www.nature.com/articles/s42256-023-00712-7)

### **5. [Large models + machine learning for high-precision prediction of enzyme kinetic parameters](https://hyper.ai/news/29000)**

- **Research highlight:** [https://hyper.ai/news/29000](https://hyper.ai/news/29000)
- **Research Team:** Xiaozhou Luo's Research Team at CAS
- **Related Research:** kcat/Km dataset, Michaelis constant dataset, pH and temperature dataset, DLKcat dataset, UniKP framework, ProtT5-XL-UniRef50, SMILES Transformer model, ensemble models, Random Forest, Extremely Randomized Trees, linear regression models
- **Published Journal:** Nature Communications, 2023.12
- **Paper Link:** [UniKP: a unified framework for the prediction of enzyme kinetic parameters](https://www.nature.com/articles/s41467-023-44113-1)

### **6. [MIT uses deep learning to discover novel antibiotics](https://hyper.ai/news/28886)**

- **Research highlight:** [https://hyper.ai/news/28886](https://hyper.ai/news/28886)
- **Research Team:** MIT Research Team
- **Related Research:** Mcule database, Broad Institute database, Graph Neural Network Chemprop, deep learning. Screened 3,646 antibiotic compounds.
- **Published Journal:** Nature, 2023.12
- **Paper Link:** [Discovery of a structural class of antibiotics with explainable deep learning](https://www.nature.com/articles/s41586-023-06887-8)

### **7. [Neural networks decipher GPCR-G protein coupling selectivity](https://hyper.ai/news/28361)**

- **Research highlight:** [https://hyper.ai/news/28361](https://hyper.ai/news/28361)
- **Research Team:** Research Team at the University of Florida
- **Related Research:** Binary classification neural networks, machine learning, unsupervised deep learning models. Built coarse-grained models of 124 GPCRs from different mammals.
- **Published Journal:** Cell Reports, 2023.09
- **Paper Link:** [Rules and mechanisms governing G protein coupling selectivity of GPCRs](https://doi.org/10.1016/j.celrep.2023.113173)

### **8. [Macformer macrocyclizes the acyclic drug fedratinib](https://hyper.ai/news/28189)**

- **Research highlight:** [https://hyper.ai/news/28189](https://hyper.ai/news/28189)
- **Research Team:** Honglin Li's Research Group at East China University of Science and Technology
- **Related Research:** ZINC dataset, ChEMBL database, deep learning models, Transformer architecture, Macformer
- **Published Journal:** Nature Communication, 2023.07
- **Paper Link:** [Macrocyclization of linear molecules by deep learning to facilitate macrocyclic drug candidates discovery](https://www.nature.com/articles/s41467-023-40219-8)

### **9. [Regression network + CGMD predicts self-assembly properties of tens of billions of peptides](https://hyper.ai/news/26408)**

- **Research highlight:** [https://hyper.ai/news/26408](https://hyper.ai/news/26408)
- **Research Team:** Wenbin Li's Research Group at Westlake University
- **Related Research:** Latin Hypercube Sampling, CGMD model, AP prediction model, Transformer, MLP, TRN model. Obtained the AP of pentapeptides and decapeptides.
- **Published Journal:** Advanced Science, 2023.09
- **Paper Link:** [Deep Learning Empowers the Discovery of Self-Assembling Peptides with Over 10 Trillion Sequences](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301544)

### **10. [Unsupervised learning predicts 71 million gene mutations](https://hyper.ai/news/26154)**

- **Research highlight:** [https://hyper.ai/news/26154](https://hyper.ai/news/26154)
- **Research Team:** Google DeepMind Research Team
- **Related Research:** ClinVar dataset, AlphaFold, weak-label learning, unsupervised learning, AlphaMissense
- **Published Journal:** Science, 2023.09
- **Paper Link:** [Accurate proteome-wide missense variant effect prediction with AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492)

### **11. [Odor analysis AI developed based on Graph Neural Networks (GNN)](https://hyper.ai/news/25952)**

- **Research highlight:** [https://hyper.ai/news/25952](https://hyper.ai/news/25952)
- **Research Team:** Osmo, a spin-off of Google Research
- **Related Research:** GS-LF database, GNN, Bayesian optimization algorithm. Outperformed humans in 53% of chemical molecules and 55% of odor descriptors judgments.
- **Published Journal:** Science, 2023.08
- **Paper Link:** [A principal odor map unifies diverse tasks in olfactory perception](https://www.science.org/doi/full/10.1126/science.ade4401)

### **12. [Graph neural networks screen for safe and highly effective anti-aging ingredients](https://hyper.ai/news/25822)**

- **Research highlight:** [https://hyper.ai/news/25822](https://hyper.ai/news/25822)
- **Research Team:** MIT Research Team
- **Related Research:** Deep learning, GNN, Convolutional Neural Networks. The true positive rate of the Chemprop model was 11.6%, higher than the 1.9% of manual screening.
- **Published Journal:** Nature Communications, 2023.05
- **Paper Link:** [Discovering small-molecule senolytics with deep neural networks](https://www.nature.com/articles/s43587-023-00415-z)

### **13. [Machine learning quantitatively analyzes dopamine release amount and location](https://hyper.ai/news/25153)**

- **Research highlight:** [https://hyper.ai/news/25153](https://hyper.ai/news/25153)
- **Research Team:** Research Team at the University of California, Berkeley
- **Related Research:** SVM, RF, machine learning. Accuracy in determining stimulation intensity reached 0.832, and accuracy for dopamine release brain region was 0.708.
- **Published Journal:** ACS Chemical Neuroscience, 2023.06
- **Paper Link:** [Identifying Neural Signatures of Dopamine Signaling with Machine Learning](https://pubs.acs.org/doi/full/10.1021/acschemneuro.3c00001)

### **14. [Machine learning discovers three anti-aging drugs](https://hyper.ai/news/24578)**

- **Research highlight:** [https://hyper.ai/news/24578](https://hyper.ai/news/24578)
- **Research Team:** Dr. James L. Kirkland and team at Mayo Clinic
- **Related Research:** Machine learning, Random Forest (RF) model, 5-fold cross-validation. Discovered senolytic drugs Ginkgetin, Periplocin, and Oleandrin.
- **Published Journal:** Nature Communications, 2023.06
- **Paper Link:** [Discovery of Senolytics using machine learning](https://www.nature.com/articles/s41467-023-39120-1)

### **15. [Deep learning screens for novel antibiotics inhibiting Acinetobacter baumannii](https://hyper.ai/news/24499)**

- **Research highlight:** [https://hyper.ai/news/24499](https://hyper.ai/news/24499)
- **Research Team:** Research Teams at McMaster University and MIT
- **Related Research:** Broad Institute's high-throughput screening sub-library, machine learning, deep learning. Screened around 7,500 molecules, discovering an antibacterial compound named abaucin.
- **Published Journal:** Nature Chemical Biology, 2023.05
- **Paper Link:** [Deep learning-guided discovery of an antibiotic targeting Acinetobacter baumannii](https://www.nature.com/articles/s41589-023-01349-8#access-options)

### **16. [Machine learning models applied to predict bioink printability](https://hyper.ai/news/24237)**

- **Research highlight:** [https://hyper.ai/news/24237](https://hyper.ai/news/24237)
- **Research Team:** Research Teams at the University of Santiago de Compostela and UCL
- **Related Research:** Machine learning models, ANN, SVM, RF, kappa, R², MAE. Accuracy reached up to 97.22%.
- **Published Journal:** International Journal of Pharmaceutics: X, 2023.12
- **Paper Link:** [Predicting pharmaceutical inkjet printing outcomes using machine learning](https://www.sciencedirect.com/science/article/pii/S2590156723000257)

### **17. [Machine learning differentiates pluripotent stem cells](https://hyper.ai/news/23940)**

- **Research highlight:** [https://hyper.ai/news/23940](https://hyper.ai/news/23940)
- **Research Team:** Yang Zhao's and Yu Zhang's Research Groups at Peking University, jointly with Yiyan Liu's Research Group at Beijing Jiaotong University
- **Related Research:** Live-cell imaging, machine learning, weakly supervised models, pix2pix deep learning model. Increased differentiation efficiency from 21.6% ± 2.7% to 88.8% ± 10.5%.
- **Published Journal:** Cell Discovery, 2023.06
- **Paper Link:** [A live-cell image-based machine learning strategy for reducing variability in PSC differentiation systems](https://www.nature.com/articles/s41421-023-00543-1)

### **18. [Machine learning model predicts drug release rate of long-acting injectables](https://hyper.ai/news/33892)**

- **Research highlight:** [https://hyper.ai/news/33892](https://hyper.ai/news/33892)
- **Research Team:** University of Toronto Research Team
- **Related Research:** MLR, Lasso, PLS, DT, RF, LGBM, XGB, AutoNGB, SVR, k-NN, NN, nested cross-validation, farthest neighbor clustering algorithm.
- **Published Journal:** Nature Communications, 2023.01
- **Paper Link:** [Machine learning models to accelerate the design of polymeric long-acting injectables](https://www.nature.com/articles/s41467-022-35343-w)

### **19. [Machine learning algorithm effectively predicts plant antimalarial properties](https://hyper.ai/news/33883)**

- **Research highlight:** [https://hyper.ai/news/33883](https://hyper.ai/news/33883)
- **Research Team:** Royal Botanic Gardens, Kew, and University of St Andrews Research Team
- **Related Research:** Logit, SVC, XGB, BNN, GridSearchCV algorithms, 10-fold stratified cross-validation, Markov Chain Monte Carlo iterations. Accuracy of 0.67.
- **Published Journal:** Frontiers in Plant Science, 2023.05
- **Paper Link:** [Machine learning enhances prediction of plants as potential sources of antimalarials](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10248027)

### **20. [Machine learning ensemble method predicts immunogenicity of viral protein fragments](https://hyper.ai/news/30786)**

- **Research highlight:** [https://hyper.ai/news/30786](https://hyper.ai/news/30786)
- **Research Team:** Jing Li's Research Team at Beihang University
- **Related Research:** Protein database UniProt, Protegen database, ensemble machine learning approach VirusImmu, RF, XGBoost, kNN, random sampling cross-validation.
- **Published Journal:** bioRxiv, 2023.11
- **Paper Link:** [VirusImmu: a novel ensemble machine learning approach for viral immunogenicity prediction](https://www.biorxiv.org/content/10.1101/2023.11.23.568426v1)

### **21. [Generative AI used to develop novel antibiotics](https://hyper.ai/news/31421)**

- **Research highlight:** [https://hyper.ai/news/31421](https://hyper.ai/news/31421)
- **Research Team:** McMaster University and Stanford University Team
- **Related Research:** Pharmakon-1760 library, Drug Repurposing Hub database, synthetic small molecule screening set, Monte Carlo Tree Search, generative AI model SyntheMol. Generated 24,335 complete molecules and designed structurally novel compounds easy to synthesize.
- **Published Journal:** Nature Machine Intelligence, 2024.03
- **Paper Link:** [Generative AI for designing and validating easily synthesizable and structurally novel antibiotics](https://www.nature.com/articles/s42256-024-00809-7 )

### **22. [Deep learning-based automated, high-speed, multidimensional single-particle tracking system](https://hyper.ai/news/31341)**

- **Research highlight:** [https://hyper.ai/news/31341](https://hyper.ai/news/31341)
- **Research Team:** Prof. Ning Fang's Team at Xiamen University
- **Related Research:** Multidimensional imaging devices, dual-focal plane imaging, parallax microscopy, multidimensional imaging equipment, Convolutional Neural Network models, noise resistance, and robustness.
- **Published Journal:** Nature Machine Intelligence, 2024.03
- **Paper Link:** [Deep Learning-Assisted Automated Multidimensional Single Particle Tracking in Living Cells](https://doi.org/10.1021/acs.nanolett.3c04870)

### **23. [ProEnsemble machine learning framework: Optimizing evolutionary pathway promoter combinations](https://hyper.ai/news/30594)**

- **Research highlight:** [https://hyper.ai/news/30594](https://hyper.ai/news/30594)
- **Research Team:** Xiaozhou Luo's Team at CAS
- **Related Research:** Synthetic biology, gene epistasis, automation platforms, 10-fold cross-validation, ensemble models, Gradient Boosting Regressor, Ridge Regressor, Gradient Boosting, universal chassis for efficient synthesis of flavonoids.
- **Published Journal:** ADVANCED SCIENCE, 2024.02
- **Paper Link:** [Pathway Evolution Through a Bottlenecking-Debottlenecking Strategy and Machine Learning-Aided Flux Balancing](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202306935)

### **24. [Microenvironment-aware graph neural network ProtLGN guides directed protein evolution](https://hyper.ai/news/32246)**

- **Research highlight:** [https://hyper.ai/news/32246](https://hyper.ai/news/32246)
- **Research Team:** Liang Hong's Research Group at Shanghai Jiao Tong University
- **Related Research:** Microenvironment-aware graph neural network, lightweight graph denoising networks, self-supervised pre-training, equivariant graph neural networks. Over 40% of PROTLGN-designed single-point mutant proteins outperformed their wild-type counterparts.
- **Published Journal:** JOURNAL OF CHEMICAL INFORMATION AND MODELING, 2024.04
- **Paper Link:** [Protein Engineering with Lightweight Graph Denoising Neural Networks](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00036)

### **25. [Deep learning model AlphaPPIMd: Exploring conformational ensembles of protein-protein complexes](https://hyper.ai/news/32435)**

- **Research highlight:** [https://hyper.ai/news/32435](https://hyper.ai/news/32435)
- **Research Team:** Jianmin Wang's Team at Yonsei University
- **Related Research:** Deep learning, generative AI, Transformer, Generative Neural Network learning, molecular dynamics, barnase-barstar complex trajectory set, Protein Data Bank, AlphaPPIMd model, self-attention mechanism, feature optimization module, attention scores, all-atom model. Average training accuracy was 0.995, and average validation accuracy was 0.999.
- **Published Journal:** Journal of Chemical Theory and Computation, 2024.05
- **Paper Link:** [Exploring the conformational ensembles of protein-protein complex with transformer-based generative model](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00255)

### **26. [Novel tumor-suppressor protein degrader dp53m inhibits cancer cell proliferation](https://hyper.ai/news/32527)**

- **Research highlight:** [https://hyper.ai/news/32527](https://hyper.ai/news/32527)
- **Research Team:** Prof. Sijin Wu's team at Xi'an Jiaotong-Liverpool University Huihu College of Pharmacy, and Prof. Songbo Xie & Prof. Diansheng Zhong's team at Tianjin Medical University General Hospital
- **Related Research:** MD simulation, iterative molecular docking-guided post-SELEX method. dp53m specifically recognizes the p53-R175H protein and degrades it.
- **Published Journal:** Science Bulletin, 2024.05
- **Paper Link:** [An engineered DNA aptamer-based PROTAC for precise therapy of p53-R175H hotspot mutant-driven cancer](https://www.sciencedirect.com/science/article/pii/S2095927324003517)

### **27. [CVPR Best Student Paper! Multimodal model BioCLIP achieves zero-shot learning](https://hyper.ai/news/32544)**

- **Research highlight:** [https://hyper.ai/news/32544](https://hyper.ai/news/32544)
- **Research Team:** Jiaman Wu's Team at The Ohio State University
- **Related Research:** Bio-image dataset TreeOfLife-10M, multimodal models, computer vision, vision encoder, text encoder, autoregressive language model. The model performed excellently in zero-shot and few-shot tasks.
- **Published Journal:** CVPR 2024, 2024.02
- **Paper Link:** [BIoCLIP: A Vision Foundation Model for the Tree of Life](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html)

### **28. [100 million parameters! Cell foundation model scFoundation models 20,000 genes simultaneously](https://hyper.ai/news/32623)**

- **Research highlight:** [https://hyper.ai/news/32623](https://hyper.ai/news/32623)
- **Research Team:** Prof. Xuegong Zhang (Tsinghua University), Prof. Jianzhu Ma (Tsinghua AIR), and Dr. Le Song (BioMap)
- **Related Research:** AI cell foundation model, human single-cell omics data DISCO, EMBL-EBI databases, GEO datasets, Single Cell Portal datasets, HCA datasets, hECA datasets, Transformer, asymmetric encoder-decoder structure, vector modules, RDA modeling.
- **Published Journal:** Nature Methods, 2024.06
- **Paper Link:** [Large-scale foundation model on single-cell transcriptomics](https://www.nature.com/articles/s41592-024-02305-7)

### **29. [Accepted by ICML, protein language model ESM-AA surpasses traditional SOTA](https://hyper.ai/news/32674)**

- **Research highlight:** [https://hyper.ai/news/32674](https://hyper.ai/news/32674)
- **Research Team:** Prof. Hao Zhou (Tsinghua University), jointly with Peking University, Nanjing University, and Shuimu BioSciences
- **Related Research:** Protein dataset AlphaFold DB, protein dataset Dp and a molecular dataset Dm, decompression, multi-scale masked language modeling.
- **Published Journal:** ICML 2024, 2024.06
- **Paper Link:** [ESM All-Atom: Multi-scale Protein Language Model for Unified Molecular Modeling](https://icml.cc/virtual/2024/poster/35119)

### **30. [SPACE algorithm published in Cell sub-journal! Tissue module discovery capabilities lead similar tools](https://hyper.ai/news/32738)**

- **Research highlight:** [https://hyper.ai/news/32738](https://hyper.ai/news/32738)
- **Research Team:** Qiangfeng Zhang's Group at Tsinghua University
- **Related Research:** Spatial transcriptomics, STARmap mouse PLA dataset, MERFISH mouse AB dataset, MERFISH mouse WB dataset, Xenium human BC dataset, CosMx human NSCLC dataset, Visium human brain dataset, encoders, proximity graph decoders, gene expression decoders, spatial proximity, self-supervised learning.
- **Published Journal:** Cell Systems, 2024.06
- **Paper Link:** [Tissue module discovery in single-cell resolution spatial transcriptomics data via cell-cell interaction-aware cell embedding](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00124-8)

### **31. [New breakthroughs based on AlphaFold reveal dynamic protein diversity](https://hyper.ai/news/33075)**

- **Research highlight:** [https://hyper.ai/news/33075](https://hyper.ai/news/33075)
- **Research Team:** MIT Research Team
- **Related Research:** Flow matching technology, protein language models, neural networks, AlphaFold, ESMFold.
- **Published Journal:** ICML 2024, 2024.06
- **Paper Link:** [AlphaFold Meets Flow Matching for Generating Protein Ensembles](https://openreview.net/forum?id=rs8Sh2UASt)

### **32. [P450Diffusion: De novo design method for P450 enzymes developed based on diffusion models](https://hyper.ai/news/33057)**

- **Research highlight:** [https://hyper.ai/news/33057](https://hyper.ai/news/33057)
- **Research Team:** Huifeng Jiang and Jian Cheng's Team at Tianjin Institute of Industrial Biotechnology, CAS
- **Related Research:** Directed evolution, diffusion models, deep learning, denoising diffusion probabilistic models, three-point anchoring, fine-tuning diffusion models, pre-training. Improved catalytic capability by 3.5 times.
- **Published Journal:** Research, 2024.07
- **Paper Link:** [Cytochrome P450 Enzyme Design by Constraining the Catalytic Pocket in a Diffusion Model](https://spj.science.org/doi/10.34133/research.0413)

### **33. [Equivariant graph neural networks used for target protein binding site prediction, boosting performance by 20%](https://hyper.ai/news/32957)**

- **Research highlight:** [https://hyper.ai/news/32957](https://hyper.ai/news/32957)
- **Research Team:** Research Team at Gaoling School of Artificial Intelligence, Renmin University of China
- **Related Research:** E(3) equivariant graph neural networks, Convolutional Neural Networks, EquiPocket framework, scPDB dataset, PDBbind dataset, COACH 420 dataset, HOLO4K dataset, local geometry modeling modules, global structural modeling modules, surface information passing modules.
- **Published Journal:** ICML 2024, 2024.07
- **Paper Link:** [EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction](https://openreview.net/forum?id=1vGN3CSxVs)

### **34. [20 experimental data points create an AI protein milestone! FSFP effectively optimizes protein pre-training models](https://hyper.ai/news/32822)**

- **Research highlight:** [https://hyper.ai/news/32822](https://hyper.ai/news/32822)
- **Research Team:** Prof. Liang Hong's Group at Shanghai Jiao Tong University, jointly with Pan Tan's Team at Shanghai Artificial Intelligence Laboratory
- **Related Research:** Protein mutation dataset ProteinGym, pre-trained protein language models, meta-transfer learning, learning to rank (LTR), parameter-efficient fine-tuning, LTR technology, FSFP training strategy, model-agnostic meta-learning methods.
- **Published Journal:** Nature Communications, 2024.07
- **Paper Link:** [Enhancing efficiency of protein language models with minimal wet-lab data through few-shot learning](https://doi.org/10.1038/s41467-024-49798-6)

### **35. [Transferable deep learning model identifies multiple types of RNA modifications, significantly reducing computational costs](https://hyper.ai/news/32745)**

- **Research highlight:** [https://hyper.ai/news/32745](https://hyper.ai/news/32745)
- **Research Team:** Assoc. Prof. Xiang Yu's Group at Shanghai Jiao Tong University, jointly with Jun Yang/Hongxia Wang's Team at Shanghai Chenshan Botanical Garden
- **Related Research:** Transferable deep learning model TandemMod, in vitro transcription dataset ELIGOS, Curlcake dataset, in vitro epitranscriptome dataset IVET, 1D CNN, Bi-LSTM modules, attention mechanisms, fully-connected classifiers.
- **Published Journal:** Nature Communications, 2024.05
- **Paper Link:** [Transfer learning enables identification of multiple types of RNA modifications using nanopore direct RNA sequencing](https://www.nature.com/articles/s41467-024-48437-4)

### **36. [InstructProtein: Aligning protein language with human language using knowledge instructions](https://hyper.ai/news/33697)**

- **Research highlight:** [https://hyper.ai/news/33697](https://hyper.ai/news/33697)
- **Research Team:** Huajun Chen and Qiang Zhang's Team at Zhejiang University
- **Related Research:** LLMs, protein knowledge instruction datasets, Gene Ontology (GO) datasets, InstructProtein, knowledge graphs, protein localization prediction, protein function prediction, protein metal-ion binding capacity prediction.
- **Published Journal:** ACL 2024, 2023.10
- **Paper Link:** [InstructProtein: Aligning Human and Protein Language via Knowledge Instruction](https://arxiv.org/abs/2310.03269)

### **37. [Protein-to-text generation framework ProtT3 enables cross-modal interpretation of protein data and text information](https://hyper.ai/news/33546)**

- **Research highlight:** [https://hyper.ai/news/33546](https://hyper.ai/news/33546)
- **Research Team:** Xiang Wang at USTC, jointly with Zhiyuan Liu's Team at NUS and Hokkaido University researchers
- **Related Research:** Cross-modal projectors, protein language models, Swiss-Prot and ProteinKG25 datasets, PDB-QA dataset.
- **Published Journal:** ACL 2024, 2023.05
- **Paper Link:** [ProtT3: Protein-to-Text Generation for Text-based Protein Understanding](https://arxiv.org/abs/2405.12564)

### **38. [CPDiffusion model designs functional proteins fully automatically at an ultra-low cost](https://hyper.ai/news/34692)**

- **Research highlight:** [https://hyper.ai/news/34692](https://hyper.ai/news/34692)
- **Research Team:** Liang Hong's Group at Shanghai Jiao Tong University
- **Related Research:** Protein engineering, diffusion probabilistic model framework CPDiffusion, amino acids, Graph Neural Networks, auxiliary drug design, protein language models, CATH 4.2 dataset.
- **Published Journal:** Cell Discovery, 2024.09
- **Paper Link:** [A conditional protein diffusion model generates artificial programmable endonuclease sequences with enhanced activity](https://www.nature.com/articles/s41421-024-00728-2)

### **39. [A novel protein homolog detection method based on protein language models and dense retrieval techniques](https://hyper.ai/news/34225)**

- **Research highlight:** [https://hyper.ai/news/34225](https://hyper.ai/news/34225)
- **Research Team:** Yu Li (CUHK), Siqi Sun (Fudan University & Shanghai AI Lab), and Mark Gerstein (Yale University)
- **Related Research:** Protein engineering, protein language models, dense retrieval techniques, dense homolog retrievers, hybrid model DHR-meta, UR90 dataset, JackHMMER algorithm, BFD/MGnify datasets, DHR method. Improved protein homolog detection sensitivity by 56%.
- **Published Journal:** Nature Biotechnology, 2024.08
- **Paper Link:** [Fast, sensitive detection of protein homologs using deep dense retrieval](https://doi.org/10.1038/s41587-024-02353-6)

### **40. [AlphaProteo efficiently designs target protein binders, increasing affinity by 300 times](https://hyper.ai/news/34214)**

- **Research highlight:** [https://hyper.ai/news/34214](https://hyper.ai/news/34214)
- **Research Team:** DeepMind, Francis Crick Institute
- **Related Research:** Protein engineering, protein language models, AI drug design, target proteins, AI tools, machine learning model AlphaProteo, VEGF-A protein binder design, Generator, Filter. Candidate binder binding was 5-100x higher than existing methods.
- **Published Journal:** DeepMind, 2024.09
- **Paper Link:** [AlphaProteo generates novel proteins for biology and health research](https://deepmind.google/discover/blog/alphaproteo-generates-novel-proteins-for-biology-and-health-research/)

### **41. [Novel denoising protein language model DePLM outperforms SOTA models in mutation effect prediction](https://hyper.ai/news/34954)**

- **Research highlight:** [https://hyper.ai/news/34954](https://hyper.ai/news/34954)
- **Research Team:** Prof. Huajun Chen and Dr. Qiang Zhang at Zhejiang University
- **Related Research:** Denoising Protein Language Model (DePLM), ProteinGym deep mutational scanning (DMS) ensemble, DMS datasets, random cross-validation, generalization experiments, extending diffusion models using sorting information to denoise evolutionary information, sorting algorithm-generated trajectories, PromptProtein model.
- **Published Journal:** NeurIPS 2024, 2024.11
- **Paper Link:** [DePLM: Denoising Protein Language Models for Property Optimization](https://neurips.cc/virtual/2024/poster/95517)

### **42. [Geometric deep generative model DynamicBind enables dynamic protein docking prediction](https://hyper.ai/news/34894)**

- **Research highlight:** [https://hyper.ai/news/34894](https://hyper.ai/news/34894)
- **Research Team:** Shuangjia Zheng's Group at Shanghai Jiao Tong University, Galixir, Sun Yat-sen University, Rice University
- **Related Research:** PDBbind dataset, MDT test set, deep diffusion models, equivariant geometric neural network technology, PDB format structures, small-molecule ligand format, contact-LDDT (cLDDT) scoring modules, AlphaFold structures, affinity prediction modules, generative AI.
- **Published Journal:** Nature Communications, 2024.02
- **Paper Link:** [DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model](https://www.nature.com/articles/s41467-024-45461-2)

### **43. [Drug discovery large language model Y-Mol completely outperforms LLaMA2](https://hyper.ai/news/35572)**

- **Research highlight:** [https://hyper.ai/news/35572](https://hyper.ai/news/35572)
- **Research Team:** Hunan University, Central South University, Hunan Normal University, Xiangtan University
- **Related Research:** Multiscale biomedical knowledge-guided LLM Y-Mol, PubMed text corpus, DrugBank benchmark dataset, DrugCentral benchmark dataset, LLaMA2-7b LLM.
- **Published Journal:** arXiv, 2024.10
- **Paper Link:** [Y-Mol: A Multiscale Biomedical Knowledge-Guided Large Language Model for Drug Development](https://doi.org/10.48550/arXiv.2410.11550)

### **44. [Universal molecular inverse folding model UniIF further complements AlphaFold 3](https://hyper.ai/news/35781)**

- **Research highlight:** [https://hyper.ai/news/35781](https://hyper.ai/news/35781)
- **Research Team:** Westlake University Future Industry Research Center Team
- **Related Research:** CATH4.3 dataset, ESM2 model, CASP15 dataset, new crystal structures, NovelPro dataset, RDesign datasets, CHILI-3K dataset, predefined frameworks based on amino acids and nucleotides, GNN, Geometric Featurizer, Block Graph Attention. Outperformed other SOTA methods in protein, RNA, and materials design.
- **Published Journal:** NeurIPS 2024, 2024.05
- **Paper Link:** [UniIF: Unified Molecule Inverse Folding](https://arxiv.org/abs/2405.18968)

### **45. [Pre-trained protein language model ProSST integrates protein structure information more effectively](https://hyper.ai/news/35874)**

- **Research highlight:** [https://hyper.ai/news/35874](https://hyper.ai/news/35874)
- **Research Team:** Prof. Liang Hong's Group and Bingxin Zhou at Shanghai Jiao Tong University, jointly with Pan Tan at Shanghai AI Lab
- **Related Research:** Pre-trained protein language model ProSST, Transformer, disentangled attention mechanisms, protein structure quantizers, AlphaFoldDB dataset, CATH43-S40 dataset, CATH43-S40 local structure dataset, ProteinGYM benchmark. Outperforms existing models in predicting thermal stability, metal-ion binding, protein localization, and GO annotations.
- **Published Journal:** NeurIPS 2024, 2024.05
- **Paper Link:** [ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention](https://neurips.cc/virtual/2024/poster/96656)

### **46. [Macrocyclic peptide binder framework RFpeptides offers new possibilities for undruggable proteins](https://hyper.ai/news/36150)**

- **Research highlight:** [https://hyper.ai/news/36150](https://hyper.ai/news/36150)
- **Research Team:** David Baker's Team at the Institute for Protein Design, UW
- **Related Research:** Diffusion model-based technology RFpeptides, utilizing modified RoseTTAFold and RFdiffusion with cyclic relative position encoding to generate precise macrocyclic backbones, drug development, AlphaFold, ProteinMPNN, Rosetta Relax. Enables targeted and efficient macrocycle design.
- **Published Journal:** bioRxiv, 2024.11
- **Paper Link:** [Accurate de novo design of high-affinity protein binding macrocycles using deep learning](https://doi.org/10.1101/2024.11.18.622547)

### **47. [Genome foundation model Evo enables prediction and generation from molecular to genome scales](https://hyper.ai/news/36266)**

- **Research highlight:** [https://hyper.ai/news/36266](https://hyper.ai/news/36266)
- **Research Team:** Stanford University and Arc Institute Research Team
- **Related Research:** Genome foundation model Evo, StripedHyena architecture. Evo can predict, generate, and design entire genome sequences.
- **Published Journal:** Science, 2024.11
- **Paper Link:** [Sequence modeling and design from molecular to genome scale with Evo](https://www.science.org/doi/10.1126/science.ado9336)

### **48. [DigFrag accurately segments molecular fragments using AI and generates 44 drug/pesticide molecules](https://hyper.ai/news/36346)**

- **Research highlight:** [https://hyper.ai/news/36346](https://hyper.ai/news/36346)
- **Research Team:** Prof. Guangfu Yang and Assoc. Prof. Fan Wang's Team at Central China Normal University
- **Related Research:** MolFrag platform, PADFrag database, graph attention mechanisms, DigFrag digital fragmentation method, DeepFMPO framework, Graph Neural Network architectures, Actor-Critic framework.
- **Published Journal:** Communications Chemistry, 2024.11
- **Paper Link:** [DigFrag as a digital fragmentation method used for artificial intelligence-based drug design](https://doi.org/10.1038/s42004-024-01346-5)

### **49. [Protein sequence large language model pre-training method PRIME](https://hyper.ai/news/36363)**

- **Research highlight:** [https://hyper.ai/news/36363](https://hyper.ai/news/36363)
- **Research Team:** Prof. Liang Hong's Group at Shanghai Jiao Tong University, Shanghai AI Lab, ShanghaiTech University, Hangzhou Medical College
- **Related Research:** Protein sequence LLM pre-training method PRIME, ProteomeAtlas database, UniProt database, ProteinGym dataset, MLM pre-training method, outperforming current SOTA methods.
- **Published Journal:** Science Advances, 2024.11
- **Paper Link:** [A General Temperature-Guided Language Model to Design Proteins of Enhanced Stability and Activity](https://www.science.org/doi/10.1126/sciadv.adr2641)

### **50. [Self-supervised deep learning method revolutionizes 3D reconstruction in cryo-electron microscopy](https://hyper.ai/news/36645)**

- **Research highlight:** [https://hyper.ai/news/36645](https://hyper.ai/news/36645)
- **Research Team:** UCLA Research Team
- **Related Research:** Self-supervised deep learning method single-particle IsoNet (spIsoNet), single-particle cryo-EM, biomacromolecule reconstruction, β-galactosidase dataset, HA trimer tilted dataset, non-symmetric ribosome datasets, HIV VLP tomography datasets, U-net architecture, anisotropy-corrected driving misalignment correction module.
- **Published Journal:** Nature Methods, 2024.11
- **Paper Link:** [Overcoming the preferred-orientation problem in cryo-EM with self-supervised deep learning](https://doi.org/10.1038/s41592-024-02505-1)

### **51. [Multimodal protein generation method PLAID generates sequences and all-atom protein structures simultaneously](https://hyper.ai/news/36750)**

- **Research highlight:** [https://hyper.ai/news/36750](https://hyper.ai/news/36750)
- **Research Team:** UC Berkeley, Microsoft Research, Genentech
- **Related Research:** Multimodal protein generation method PLAID (Protein Latent Induced Diffusion), Pfam database, ESMFold latent space, latent diffusion training, DiT block architecture, Diffusion Transformer (DiT), ESMFold model.
- **Published Journal:** ICLR 2025, 2024.12
- **Paper Link:** [Generating All-Atom Protein Structure from Sequence-Only Training Data](https://www.biorxiv.org/content/10.1101/2024.12.02.626353v1)

### **52. [Targeted molecular optimization method MOLRL based on latent reinforcement learning](https://hyper.ai/news/37285)**

- **Research highlight:** [https://hyper.ai/news/37285](https://hyper.ai/news/37285)
- **Research Team:** Researchers from Cellarity and NVIDIA
- **Related Research:** Novel targeted molecular optimization method MOLRL based on latent reinforcement learning, drug discovery tasks, Proximal Policy Optimization (PPO), Variational Autoencoders (VAE), Autoencoder (MolMIM), reaching up to 100% success rates.
- **Published Journal:** ChemRxiv, 2025.01
- **Paper Link:** [Targeted Molecular Generation With Latent Reinforcement Learning](https://go.hyper.ai/H4JhR)

### **53. [Viral variation driver prediction framework E2VD predicts evolutionary directions for COVID-19/HIV/Influenza viruses](https://hyper.ai/news/37405)**

- **Research highlight:** [https://hyper.ai/news/37405](https://hyper.ai/news/37405)
- **Research Team:** Prof. Yonghong Tian and Assoc. Prof. Jie Chen at Peking University, Researcher Peng Zhou at Guangzhou Laboratory
- **Related Research:** Viral variation driver prediction framework E2VD, UniRef90 dataset, open-source deep mutational scanning datasets, protein sequence encoding, Local-global dependence coupling, multi-task focal learning. Increased prediction accuracy by 67%.
- **Published Journal:** Nature Machine Intelligence, 2025.01
- **Paper Link:** [A unified evolution-driven deep learning framework for virus variation driver prediction](https://www.nature.com/articles/s42256-024-00966-9)

### **54. [Medical language model MedFound approaches expert physician reasoning capabilities](https://hyper.ai/news/37646)**

- **Research highlight:** [https://hyper.ai/news/37646](https://hyper.ai/news/37646)
- **Research Team:** Interdisciplinary team led by Prof. Guangyu Wang (BUPT), Prof. Chunli Song (Peking University Third Hospital), and Prof. Jian Yang (China Three Gorges University)
- **Related Research:** LLM BLOOM-176B, medical corpus dataset MedCorpus, medical LLM MedFound-DX, chain-of-thought methods, preference alignment framework, MedDX-FT dataset, MedDX-Bench dataset.
- **Published Journal:** Nature Medicine, 2025.01
- **Paper Link:** [A generalist medical language model for disease diagnosis assistance](https://www.nature.com/articles/s41591-024-03416-6)

### **55. [4D diffusion model AlphaFolding fills the gap in dynamic protein structure prediction](https://hyper.ai/news/37697)**

- **Research highlight:** [https://hyper.ai/news/37697](https://hyper.ai/news/37697)
- **Research Team:** Prof. Siyu Zhu and Prof. Yuan Qi's Teams at Fudan University/Shanghai AI Lab, jointly with Prof. Yao Yao at Nanjing University
- **Related Research:** 4D diffusion model AlphaFolding, MD simulation data, dynamic protein structures, structural biology, Distributional Graphformer (DiG) deep learning framework, ATLAS dataset.
- **Published Journal:** arXiv, 2024.12
- **Paper Link:** [4D Diffusion for Dynamic Protein Structure Prediction with Reference and Motion Guidance](https://arxiv.org/abs/2408.12419)

### **56. [PepPrCLIP pipeline for designing short proteins holds promise for developing new cancer therapies](https://hyper.ai/news/37912)**

- **Research highlight:** [https://hyper.ai/news/37912](https://hyper.ai/news/37912)
- **Research Team:** Duke University Biomedical Engineering Team
- **Related Research:** ESM-2 protein language model, ESM-2-650M model, PepPrCLIP pipeline, Gaussian distributions, amino acid sequences.
- **Published Journal:** Science Advances, 2025.01
- **Paper Link:** [De novo design of peptide binders to conformationally diverse targets with contrastive language modeling](https://www.science.org/doi/10.1126/sciadv.adr8638)

### **57. [Boltzmann alignment technique drastically improves protein binding free energy prediction efficacy](https://hyper.ai/news/38092)**

- **Research highlight:** [https://hyper.ai/news/38092](https://hyper.ai/news/38092)
- **Research Team:** Prof. Chunhua Shen's Team at Zhejiang University, University of Adelaide, Northeastern University (US)
- **Related Research:** Binding free energy, Boltzmann alignment technique, ∆∆G prediction, protein complex structure prediction, Riemannian diffusion models, deep learning, BA-Cycle method, BA-DDG method, SKEMPI v2 dataset.
- **Published Journal:** ICLR 2025, 2024.10
- **Paper Link:** [Boltzmann-Aligned Inverse Folding Model as a Predictor of Mutational Effects on Protein-Protein Interactions](https://arxiv.org/abs/2410.09543)

### **58. [Novel large-scale flow-based protein backbone generator Proteina achieves SOTA in de novo protein backbone design](https://hyper.ai/news/38120)**

- **Research highlight:** [https://hyper.ai/news/38120](https://hyper.ai/news/38120)
- **Research Team:** NVIDIA, Mila, University of Montreal, MIT
- **Related Research:** Protein design, scalable non-equivariant Transformer architectures, Foldseek AFDB clustered DFS dataset, D21M dataset, MFS model, staged training strategies.
- **Published Journal:** ICLR 2025 Oral, 2025.01
- **Paper Link:** [Proteina: Scaling Flow-based Protein Structure Generative Models](https://openreview.net/forum?id=TVQLu34bdw&nesting=2&sort=date-desc)

### **59. [UniGEM model achieves synergistic enhancement of two tasks based on diffusion models for the first time](https://hyper.ai/news/38186)**

- **Research highlight:** [https://hyper.ai/news/38186](https://hyper.ai/news/38186)
- **Research Team:** Tsinghua University, Chinese Academy of Sciences
- **Related Research:** Drug discovery, molecular property prediction, molecule generation, diffusion models, QM9 dataset, GEOM-Drugs 3D molecular conformation dataset, multi-task learning frameworks, E(3) Equivariant Diffusion Models (EDM), multi-branch network architectures.
- **Published Journal:** ICLR 2025, 2025.04
- **Paper Link:** [UniGEM: A Unified Approach to Generation and Property Prediction for Molecules](https://openreview.net/pdf?id=Lb91pXwZMR)

### **60. [RFdiffusion evolves further, realizing atomic-accuracy de novo antibody design](https://hyper.ai/news/38253)**

- **Research highlight:** [https://hyper.ai/news/38253](https://hyper.ai/news/38253)
- **Research Team:** Prof. David Baker's Team at University of Washington and collaborators
- **Related Research:** Therapeutic antibodies, RFdiffusion network for computational protein design, antibody variable heavy chains (VHHs), single-chain variable fragments (scFvs), deep learning, VHH frameworks, CDR loop sequence design.
- **Published Journal:** bioRxiv, 2025.02
- **Paper Link:** [Atomically accurate de novo design of antibodies with RFdiffusion](https://doi.org/10.1101/2024.03.14.585103)

### **61. [First protein-RNA language model fusion scheme sets new SOTA in binding affinity prediction](https://hyper.ai/news/38290)**

- **Research highlight:** [https://hyper.ai/news/38290](https://hyper.ai/news/38290)
- **Research Team:** Tsinghua University, UCL, Monash University, BUPT
- **Related Research:** Protein-RNA, CoPRA model, Protein Language Models (PLM), RNA Language Models (RLM), CLIP experimental techniques, Co-Former model, PDBbind dataset, PRBABv2 dataset, ProNAB dataset, PRA201 dataset, multimodal learning.
- **Published Journal:** AAAI 2025, 2025.01
- **Paper Link:** [CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction](https://arxiv.org/abs/2409.03773)

### **62. [Virtual tissue model Celcomen achieves causal inference identifiability in spatial transcriptomics analysis for the first time](https://hyper.ai/news/38308)**

- **Research highlight:** [https://hyper.ai/news/38308](https://hyper.ai/news/38308)
- **Research Team:** University of Cambridge
- **Related Research:** Perturbmap dataset, fetal spleen dataset, glioblastoma dataset, Celcomen model, inference modules (CCE), generative modules (SCE), Graph Neural Networks.
- **Published Journal:** ICLR 2025, 2025.01
- **Paper Link:** [Estimation of single-cell and tissue perturbation effect in spatial transcriptomics via Spatial Causal Disentanglement](https://openreview.net/forum?id=Tqdsruwyac)

### **63. [AlphaFold-Metainference method accurately predicts disordered protein structural ensembles](https://hyper.ai/news/38448)**

- **Research highlight:** [https://hyper.ai/news/38448](https://hyper.ai/news/38448)
- **Research Team:** University of Cambridge
- **Related Research:** Alignment error maps predicted by AlphaFold, correlations between distance variation matrices in MD simulations, disordered protein structure prediction, Protein Data Bank (PDB), Small-Angle X-ray Scattering (SAXS) data, NMR measurements, Aβ and α-synuclein structural ensembles, CALVADOS-2, Bayesian metainference methods, Langevin integrators.
- **Published Journal:** Nature Communications, 2025.02
- **Paper Link:** [AlphaFold prediction of structural ensembles of disordered proteins](https://www.nature.com/articles/s41467-025-56572-9)

### **64. [High-accuracy RNA structure prediction framework DRfold2 surpasses SOTA in multiple benchmarks](https://hyper.ai/news/38506)**

- **Research highlight:** [https://hyper.ai/news/38506](https://hyper.ai/news/38506)
- **Research Team:** Prof. Yang Zhang's Team at NUS
- **Related Research:** RNA structure prediction framework DRfold2, unsupervised contact prediction accuracy, composite RNA language models, DRfold2 RNA test datasets, CASP15 dataset, Transformer modules, denoising structural modules.
- **Published Journal:** bioRxiv, 2025.03
- **Paper Link:** [Ab initio RNA structure prediction with composite language model and denoised end-to-end learning](https://www.biorxiv.org/content/10.1101/2025.03.05.641632v1)

### **65. [New protein design algorithm DRAKES breaks through the biological sequence design bottleneck](https://hyper.ai/news/38675)**

- **Research highlight:** [https://hyper.ai/news/38675](https://hyper.ai/news/38675)
- **Research Team:** Researchers from MIT, Harvard, Stanford, UC Berkeley, Genentech
- **Related Research:** Reinforcement learning frameworks, PDB training sets, Megascale dataset, DRAKES algorithm, Gumbel-Softmax.
- **Published Journal:** ICLR 2025, 2024.08
- **Paper Link:** [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://doi.org/10.48550/arXiv.2410.13643)

### **66. [Machine learning-assisted UV absorbance spectroscopy for detecting microbial contamination](https://hyper.ai/news/38869)**

- **Research highlight:** [https://hyper.ai/news/38869](https://hyper.ai/news/38869)
- **Research Team:** SMART (Singapore-MIT Alliance for Research and Technology), A*SRL Singapore, NUS, MIT
- **Related Research:** Microbial contamination detection, anomaly detection strategies, machine learning, Support Vector Machines (SVM), radial basis functions, PBS sterilized samples.
- **Published Journal:** Nature, 2025.03
- **Paper Link:** [Machine learning aided UV absorbance spectroscopy for microbial contamination in cell therapy products](https://hyper.ai/en/sota/papers/s41598-024-83114-y)

### **67. [Utilizing protein sequence generative models for overlapping gene design](https://hyper.ai/news/39241)**

- **Research highlight:** [https://hyper.ai/news/39241](https://hyper.ai/news/39241)
- **Research Team:** David Baker's Team at University of Washington
- **Related Research:** Overlapping genes (OLG), synthetic OLG design research, amino acid substitution, bioinformatics screening, statistical modeling, systematic scanning of sequence positions.
- **Published Journal:** bioRxiv, 2025.05
- **Paper Link:** [Design of overlapping genes using deep generative models of protein sequences](https://doi.org/10.1101/2025.05.06.652464)

### **68. [Prediction framework PUPS enables single-cell level protein subcellular localization](https://hyper.ai/news/39549)**

- **Research highlight:** [https://hyper.ai/news/39549](https://hyper.ai/news/39549)
- **Research Team:** MIT, Harvard University
- **Related Research:** Protein subcellular localization, Human Protein Atlas, unseen protein subcellular localization, Predictions of Unseen Proteins’ Subcellular localization (PUPS) framework, held-out datasets, ESM-2 protein language models, CNNs, separable convolutions.
- **Published Journal:** Nature Methods, 2025.05
- **Paper Link:** [Prediction of protein subcellular localization in single cells](https://go.hyper.ai/LeaQF)

### **69. [UniMoMo: The first unified generative framework across molecular species enables multi-type drug molecular design](https://hyper.ai/news/39852)**

- **Research highlight:** [https://hyper.ai/news/39852](https://hyper.ai/news/39852)
- **Research Team:** Yang Liu's Group (Tsinghua), Wenbing Huang's Group (Renmin University), ByteDance AI Drug Discovery Team
- **Related Research:** UniMoMo framework, All-atom Iterative Variational Autoencoder (IterVAE), all-atom geometric latent space diffusion models, unified modeling.
- **Published Journal:** ICML 2025, 2025.03
- **Paper Link:** [UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](https://go.hyper.ai/LeaQF)

### **70. [Protein language model Prot42 generates high-affinity binders using only the target protein sequence](https://hyper.ai/news/40385)**

- **Research highlight:** [https://hyper.ai/news/40385](https://hyper.ai/news/40385)
- **Research Team:** Inception AI (Abu Dhabi, UAE) and Cerebras Systems (Silicon Valley, USA)
- **Related Research:** PDIdb 2010 dataset, UniRef50 database, STRING database, protein function prediction, protein subcellular localization prediction, protein structure prediction, PPI prediction, protein binder generation, DNA sequence-specific binder generation.
- **Published Journal:** arXiv, 2025.05
- **Paper Link:** [Prot42: a Novel Family of Protein Language Models for Target-aware Protein Binder Generation](https://go.hyper.ai/cFupD)

### **71. [Unified biomolecular dynamics simulator UniSim achieves unified time-coarsened dynamics simulation across molecular types and chemical environments for the first time](https://hyper.ai/news/40483)**

- **Research highlight:** [https://hyper.ai/news/40483](https://hyper.ai/news/40483)
- **Research Team:** Yang Liu's Group (Tsinghua) and Wenbing Huang's Group (Renmin University)
- **Related Research:** Atomic embedding expansion, multi-head hybrid pre-training, TorchMD-NET GNN models, stochastic interpolant frameworks, force-guided kernels.
- **Published Journal:** ICML 2025, 2025.05
- **Paper Link:** [UniSim: A Unified Simulator for Time-Coarsened Dynamics of Biomolecules](https://go.hyper.ai/5NWuO)

### **72. [Computational biology algorithm SimplifiedBondfinder uncovers 69 novel nitrogen-oxygen-sulfur bonds](https://hyper.ai/news/40515)**

- **Research highlight:** [https://hyper.ai/news/40515](https://hyper.ai/news/40515)
- **Research Team:** Sophia Bazzi and Sharareh Sayyad's Team at University of Göttingen
- **Related Research:** SimplifiedBondfinder algorithm, machine learning, quantum mechanical calculations, PDB dataset, PDB-REDO dataset, BDB dataset, UMAP dimensionality reduction, NOS linkages.
- **Published Journal:** Communications Chemistry, 2025.05
- **Paper Link:** [Revealing arginine-cysteine and glycine-cysteine NOS linkages by a systematic re-evaluation of protein structures](https://www.nature.com/articles/s42004-025-01535-w)

### **73. [Novel protein sequence design method FAMPNN simultaneously processes protein backbone and sidechain information](https://hyper.ai/news/41545)**

- **Research highlight:** [https://hyper.ai/news/41545](https://hyper.ai/news/41545)
- **Research Team:** Stanford University, Arc Institute (Palo Alto)
- **Related Research:** Protein sidechain conformations, FAMPNN method, S40 dataset, PDB dataset, CASP13/14/15 datasets, SKEMPlv2 dataset, S669 dataset, Megascale dataset, FireProtDB dataset, CR9114/CR6261 datasets, iterative sampling strategies, atom37 formats, GNNs, token-wise Euclidean diffusion methods.
- **Published Journal:** ICML 2025, 2025.06
- **Paper Link:** [Sidechain conditioning and modeling for full-atom protein sequence design with FAMPNN](https://go.hyper.ai/JUJDq)

### **74. [Atomistic protein design method La-Proteina generates proteins with up to 800 residues at high precision](https://hyper.ai/news/41744)**

- **Research highlight:** [https://hyper.ai/news/41744](https://hyper.ai/news/41744)
- **Research Team:** NVIDIA, Mila
- **Related Research:** Atomistic protein design, partially latent flow matching framework La-Proteina, AFDB dataset, two-stage training strategy.
- **Published Journal:** arXiv, 2025.06
- **Paper Link:** [La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching](https://go.hyper.ai/3csT5)

### **75. [APM model specifically designed for multi-chain protein complexes enables all-atom design and functional optimization](https://hyper.ai/news/42059)**

- **Research highlight:** [https://hyper.ai/news/42059](https://hyper.ai/news/42059)
- **Research Team:** Hunan University, UCAS, ByteDance Seed Team
- **Related Research:** Proteins, multi-chain native modeling, all-atom representation optimization, sequence-structure dependency reinforcement, PDB database, Swiss-Prot database, AFDB database, multi-chain protein datasets.
- **Published Journal:** ICML 2025, 2025.07
- **Paper Link:** [An All-Atom Generative Model for Designing Protein Complexes](https://go.hyper.ai/TVp4i)

### **76. [New intrinsically disordered region-binding protein design method Logos specializes in undruggable targets](https://hyper.ai/news/42611)**

- **Research highlight:** [https://hyper.ai/news/42611](https://hyper.ai/news/42611)
- **Research Team:** David Baker's Team at University of Washington
- **Related Research:** RFdiffusion model, Induced Fit, Scaffold Generation, Pocket Specialization, Pocket Assembly.
- **Published Journal:** Science, 2025.07
- **Paper Link:** [Design of intrinsically disordered region binding proteins](https://www.science.org/doi/10.1126/science.adr8063)

### **77. [Novel protein dynamic fusion representation framework FusionProt released, enabling iterative information exchange](https://hyper.ai/news/43724)**

- **Research highlight:** [https://hyper.ai/news/43724](https://hyper.ai/news/43724)
- **Research Team:** Technion, Meta AI
- **Related Research:** Protein language models, representation learning framework FusionProt, AlphaFold DB, AlphaFold2, DeepFRI dataset, learnable fusion tokens, Multiview Contrastive learning.
- **Published Journal:** bioRxiv, 2025.08
- **Paper Link:** [FusionProt: Fusing Sequence and Structural Information for Unified Protein Representation Learning](https://go.hyper.ai/OXLYl)

### **78. [Transcriptome-guided diffusion model MorphDiff released to accelerate phenotypic drug discovery](https://hyper.ai/news/43849)**

- **Research highlight:** [https://hyper.ai/news/43849](https://hyper.ai/news/43849)
- **Research Team:** CUHK, Mohamed bin Zayed University of Artificial Intelligence
- **Related Research:** Cell morphology, Latent Diffusion Model (LDM), large-scale cell morphology image datasets, JUMP dataset, CDRP dataset, LINCS dataset, morphological VAE, latent diffusion models.
- **Published Journal:** Nature Communications, 2025.09
- **Paper Link:** [Prediction of cellular morphology changes under perturbations with a transcriptome-guided diffusion model](https://www.nature.com/articles/s41467-025-63478-z)

### **79. [AlphaPPIMI framework significantly enhances generalization, surpassing existing methods in PPI interface modulator prediction](https://hyper.ai/news/43916)**

- **Research highlight:** [https://hyper.ai/news/43916](https://hyper.ai/news/43916)
- **Research Team:** China University of Petroleum, Yonsei University
- **Related Research:** Protein-protein interactions, DLiP dataset, ECFP4 fingerprints, ChemDiv database, AlphaPPIMI framework, Uni-Mol2 model, protein feature extraction, Transformer architecture, ESM2-150M model, ProtTrans model.
- **Published Journal:** Journal of Cheminformatics, 2025.08
- **Paper Link:** [Alphappimi: a comprehensive deep learning framework for predicting PPI-modulator interactions](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-01077-2)

### **80. [A novel fusion neural network framework efficiently predicts multi-metal binding sites in protein sequences](https://hyper.ai/news/44702)**

- **Research highlight:** [https://hyper.ai/news/44702](https://hyper.ai/news/44702)
- **Research Team:** Hong Kong University of Science and Technology
- **Related Research:** Fusion neural network framework, protein sequence multi-metal binding site prediction, CNNs, fusion networks, MbPA database, deep learning frameworks.
- **Published Journal:** bioRxiv, 2025.09
- **Paper Link:** [A Modular Fusion Neural Network Approach to Efficiently Predict Multi-Metal Binding Sites in Protein Sequences](https://go.hyper.ai/Y7DNU)

### **81. [Highly synthesizable molecular projection framework ReaSyn released, achieving ultra-high reconstruction rates and pathway diversity](https://hyper.ai/news/44764)**

- **Research highlight:** [https://hyper.ai/news/44764](https://hyper.ai/news/44764)
- **Research Team:** NVIDIA Research Team
- **Related Research:** Drug discovery, ReaSyn framework, supervised learning, reinforcement learning fine-tuning, Transformer models, Chain-of-Reaction (CoR) representation.
- **Published Journal:** arXiv, 2025.09
- **Paper Link:** [Rethinking Molecule Synthesizability with Chain-of-Reaction](https://arxiv.org/abs/2509.16084)

### **82. [Constrained reinforcement learning framework Ctrl-DNA released, realizing "targeted control" of specific cell gene expression](https://hyper.ai/news/45227)**

- **Research highlight:** [https://hyper.ai/news/45227](https://hyper.ai/news/45227)
- **Research Team:** University of Toronto Team, Changping Laboratory
- **Related Research:** Constrained RL framework Ctrl-DNA, deep learning, cell-specific gene expression, DNA language models, human promoter datasets, enhancer datasets, controllable cell-type specific CRE generation, Constrained Markov Decision Processes, Enformer architecture.
- **Published Journal:** NeurIPS 2025, 2025.05
- **Paper Link:** [Ctrl-DNA: Constrained Reinforcement Learning for Cell-Specific Cis-Regulatory Element Design](https://arxiv.org/abs/2505.20578)

### **83. [PLACER framework resolves the atomic-level modeling challenge of protein conformational heterogeneity](https://hyper.ai/news/46009)**

- **Research highlight:** [https://hyper.ai/news/46009](https://hyper.ai/news/46009)
- **Research Team:** Prof. David Baker's Research Team
- **Related Research:** Graph Neural Network PLACER, Cambridge Structural Database, PDB, denoising neural networks, 3-track architectures, small-molecule structural generation.
- **Published Journal:** PNAS, 2025.11
- **Paper Link:** [Modeling protein-small molecule conformational ensembles with PLACER](https://www.biorxiv.org/content/10.1101/2024.09.25.614868v2)

### **84. [Squidiff enables multi-scenario transcriptome simulation, boosting precision medicine and spatial medicine development](https://hyper.ai/news/46212)**

- **Research highlight:** [https://hyper.ai/news/46212](https://hyper.ai/news/46212)
- **Research Team:** Columbia University, Stanford University
- **Related Research:** Squidiff framework, Splatter tools, human iPSC-to-endoderm differentiation datasets, K562 CRISPR screening experiments, conditional DDIM, semantic encoding techniques, Encode-Diffuse-Decode architectures.
- **Published Journal:** Nature Methods, 2025.11
- **Paper Link:** [Squidiff: predicting cellular development and responses to perturbations using a diffusion model](https://www.nature.com/articles/s41592-025-02877-y)

### **85. [Generative model PepTron and new evaluation benchmark released, reshaping prediction capabilities for disordered protein ensembles](https://hyper.ai/news/47063)**

- **Research highlight:** [https://hyper.ai/news/47063](https://hyper.ai/news/47063)
- **Research Team:** Peptone, University of Copenhagen, NVIDIA, Oxford University, MIT, Duke University
- **Related Research:** PeptoneBench evaluation framework, generative model PepTron, PDB, IDRome database, NVIDIA BioNeMo, ESMFlow, mixed training strategies (experimental + synthetic data).
- **Published Journal:** bioRxiv, 2025.10
- **Paper Link:** [Advancing Protein Ensemble Predictions Across the Order–Disorder Continuum](https://www.biorxiv.org/content/10.1101/2025.10.18.680935v1)

### **86. [MIT and Harvard propose end-to-end AI workflow CleaveNet to overcome highly specific protease substrate design challenges](https://hyper.ai/news/48608)**

- **Research highlight:** [https://hyper.ai/news/48608](https://hyper.ai/news/48608)
- **Research Team:** Joint Team from MIT and Harvard University
- **Related Research:** Protease substrate design, CleaveNet workflow, synthetic peptides, prediction models and generative models.
- **Published Journal:** Nature Communications
- **Paper Link:** [CleaveNet: An AI-based end-to-end design workflow for protease substrates](https://www.nature.com/articles/s41467-025-67226-1)

### **87. [Goethe University Frankfurt team proposes a multi-scale classification framework to decode the complexity of the human E3 ligome](https://hyper.ai/news/48813)**

- **Research highlight:** [https://hyper.ai/news/48813](https://hyper.ai/news/48813)
- **Research Team:** Goethe University Frankfurt Research Team
- **Related Research:** Ubiquitin-proteasome system (UPS), E3 ubiquitin ligases, human E3 ligome, metric-learning.
- **Published Journal:** Nature Communications
- **Paper Link:** [Multi-scale classification decodes the complexity of the human E3 ligome](https://www.nature.com/articles/s41467-025-67450-9)

### **88. [Basecamp and NVIDIA jointly release the EDEN foundation model, enabling AI-programmable therapeutic design](https://hyper.ai/news/48964)**

- **Research highlight:** [https://hyper.ai/news/48964](https://hyper.ai/news/48964)
- **Research Team:** Basecamp Research, NVIDIA, and top academic institutions
- **Related Research:** Programmable biology, EDEN metagenomic foundation models, gene therapies, recombinases, antimicrobial peptide design.
- **Published Journal:** bioRxiv
- **Paper Link:** [Designing AI-programmable therapeutics with the EDEN family of foundation models](https://doi.org/10.64898/2026.01.12.699009)

### **89. [Microsoft and others propose the multimodal AI framework GigaTIME to generate virtual mIF atlases from routine pathology slides](https://hyper.ai/news/49359)**

- **Research highlight:** [https://hyper.ai/news/49359](https://hyper.ai/news/49359)
- **Research Team:** Microsoft Research, University of Washington, Providence Genomics
- **Related Research:** Tumor microenvironment, H&E staining, multiplex immunofluorescence (mIF), GigaTIME framework, spatial proteomics.
- **Published Journal:** Cell
- **Paper Link:** [Multimodal AI generates virtual population for tumor microenvironment modeling](https://www.cell.com/cell/fulltext/S0092-8674(25)01312-1)

### **90. [MIT proposes deep learning language model Pichia-CLM to optimize codons for enhanced recombinant protein yield](https://hyper.ai/news/49613)**

- **Research highlight:** [https://hyper.ai/news/49613](https://hyper.ai/news/49613)
- **Research Team:** MIT Research Team
- **Related Research:** Komagataella phaffii, codon optimization, Codon Usage Bias (CUB), Pichia-CLM language model, recombinant protein expression.
- **Published Journal:** PNAS
- **Paper Link:** [Pichia-CLM: A language model–based codon optimization pipeline for Komagataella phaffii](https://www.pnas.org/doi/10.1073/pnas.2522052123)

### **91. [MIT and ETH jointly propose deep learning framework APOLLO to efficiently integrate and disentangle single-cell multimodal data](https://hyper.ai/news/49702)**

- **Research highlight:** [https://hyper.ai/news/49702](https://hyper.ai/news/49702)
- **Research Team:** Joint Team from MIT and ETH Zurich
- **Related Research:** Single-cell biology, multimodal data integration, APOLLO framework, scRNA-seq, scATAC-seq, spatial morphology.
- **Published Journal:** Nature Computational Science
- **Paper Link:** [Partially shared multi-modal embedding learns holistic representation of cell state](https://www.nature.com/articles/s43588-025-00948-w)

### **92. [CUHK and others jointly propose the Bi-TEAM framework for unified cross-scale representation learning of modified peptides](https://hyper.ai/news/49833)**

- **Research highlight:** [https://hyper.ai/news/49833](https://hyper.ai/news/49833)
- **Research Team:** CUHK, Macao Polytechnic University, Zhejiang University, Second Xiangya Hospital of CSU, UESTC
- **Related Research:** Peptide structure and function modeling, non-canonical amino acid modifications, cross-scale representation learning, Bi-TEAM framework.
- **Published Journal:** arXiv
- **Paper Link:** [Bi-TEAM: A Unified Cross-Scale Representation Learning Framework for Chemically Modified Biomolecules](https://arxiv.org/abs/2603.01873)

### **93. [Carnegie Mellon University and others propose AQuaRef for quantum refinement of all-atom protein models](https://hyper.ai/news/49895)**

- **Research highlight:** [https://hyper.ai/news/49895](https://hyper.ai/news/49895)
- **Research Team:** CMU, University of Wrocław, University of Florida
- **Related Research:** Protein structure refinement, AQuaRef, machine learning interatomic potentials (AIMNet2), quantum refinement, structural biology.
- **Published Journal:** Nature Communications
- **Paper Link:** [AQuaRef: machine learning accelerated quantum refinement of protein structures](https://www.nature.com/articles/s41467-025-64313-1)

### **94. [NVIDIA and others jointly propose the Complexa framework to unify protein binder generation and optimization](https://hyper.ai/news/49977)**

- **Research highlight:** [https://hyper.ai/news/49977](https://hyper.ai/news/49977)
- **Research Team:** NVIDIA, Oxford University, Mila
- **Related Research:** Protein binder design, Proteína-Complexa (Complexa), Teddymer, generative methods, Test-Time Compute.
- **Published Journal:** ICLR 2026
- **Paper Link:** [Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute](https://openreview.net/forum?id=qmCpJtFZra)

### **95. [MIT and CMU jointly propose VibeGen, introducing vibrational dynamics to empower de novo protein design](https://hyper.ai/news/50061)**

- **Research highlight:** [https://hyper.ai/news/50061](https://hyper.ai/news/50061)
- **Research Team:** Joint Team from MIT and CMU
- **Related Research:** Protein dynamics, VibeGen agent, language diffusion models, de novo protein design, vibrational amplitude prediction.
- **Published Journal:** Matter
- **Paper Link:** [VibeGen: Agentic end-to-end de novo protein design for tailored dynamics using a language diffusion model](https://www.cell.com/matter/abstract/S2590-2385(26)00069-X)

### **96. [Institut Pasteur uses deep learning to predict 2.39 million anti-phage proteins, mapping bacterial immunity](https://hyper.ai/news/50491)**

- **Research highlight:** [https://hyper.ai/news/50491](https://hyper.ai/news/50491)
- **Research Team:** Institut Pasteur Research Team
- **Related Research:** Bacterial anti-viral immunity, anti-phage defense systems, protein language models, genomic language models, pangenomics.
- **Published Journal:** Science
- **Paper Link:** [Protein and genomic language models uncover the unexplored diversity of bacterial immunity](https://www.science.org/doi/10.1126/science.adv8275)

### **97. [KAIST team utilizes AI to de novo design small-molecule binding proteins, successfully applying them in biosensors](https://hyper.ai/news/50599)**

- **Research highlight:** [https://hyper.ai/news/50599](https://hyper.ai/news/50599)
- **Research Team:** Department of Biological Sciences Research Team at KAIST
- **Related Research:** De novo protein design, small-molecule binding proteins, NTF2-like fold, biosensors, chemically induced dimerization (CID).
- **Published Journal:** Nature Communications
- **Paper Link:** [Small-molecule binding and sensing with a designed protein family](https://www.nature.com/articles/s41467-026-70953-8)

### **98. [University of Toronto and others propose dnaHNet for efficient hierarchical modeling of genomic sequences](https://hyper.ai/news/50709)**

- **Research highlight:** [https://hyper.ai/news/50709](https://hyper.ai/news/50709)
- **Research Team:** University of Toronto, Vector Institute, Arc Institute
- **Related Research:** Genomic sequence learning, foundation models, dnaHNet, dynamic tokenization, variant effect prediction.
- **Published Journal:** arXiv
- **Paper Link:** [dnaHNet: A Scalable and Hierarchical Foundation Model for Genomic Sequence Learning](https://arxiv.org/abs/2602.10603)

### **99. [Queen Mary University of London and others conduct the largest-scale proteogenomic study, revealing molecular disease mechanisms](https://hyper.ai/news/51343)**

- **Research highlight:** [https://hyper.ai/news/51343](https://hyper.ai/news/51343)
- **Research Team:** Queen Mary University of London, Cambridge University
- **Related Research:** Proteogenomics, protein quantitative trait loci (pQTLs), circulating protein abundance, cis- and trans-genetic regulation.
- **Published Journal:** Cell
- **Paper Link:** [Multi-cohort proteogenomic analyses reveal genetic effects across the proteome and diseasome](https://www.cell.com/cell/fulltext/S0092-8674(26)00385-5)

### **100. [Goethe University Frankfurt and others propose genESOM model: Generative AI breaks through small-sample animal experiments](https://hyper.ai/news/51430)**

- **Research highlight:** [https://hyper.ai/news/51430](https://hyper.ai/news/51430)
- **Research Team:** Goethe University Frankfurt and Fraunhofer ITMP
- **Related Research:** Small-sample animal experiments, Generative AI, genESOM model, emergent self-organizing maps.
- **Published Journal:** Pharmacological Research
- **Paper Link:** [Self-organizing neural network-based generative AI with embedded error inflation control enhances effective knowledge extraction from preclinical studies with reduced sample size](https://www.sciencedirect.com/science/article/pii/S1043661826000745)

## **AI+ Healthcare**

### **1. [DeepDR Plus deep learning system predicts diabetic retinopathy using fundus images](https://hyper.ai/news/29769)**

- **Research highlight:** [https://hyper.ai/news/29769](https://hyper.ai/news/29769)
- **Research Team:** Prof. Weiping Jia, Huating Li, and Bin Sheng's Team at Shanghai Jiao Tong University; Tianyin Huang's Research Team at Tsinghua University
- **Related Research:** SDPP data, DRPS data, ResNet-50, fundus models, self-supervised learning, IBS evaluation models, meta-models. Extended the average clinical screening interval from 12 months to 31.97 months.
- **Published Journal:** Nature Medicine, 2024.01
- **Paper Link:** [A deep learning system for predicting time to progression of diabetic retinopathy](https://www.nature.com/articles/s41591-023-02702-z)

### **2. [Logistic regression model analyzes that high green landscape index reduces MetS risk](https://hyper.ai/news/29559)**

- **Research highlight:** [https://hyper.ai/news/29559](https://hyper.ai/news/29559)
- **Research Team:** Xifeng Wu's Research Team at Zhejiang University
- **Related Research:** Convolutional neural network models, logistic regression models, Isochrone API
- **Published Journal:** Environment International, 2024.01
- **Paper Link:** [Beneficial associations between outdoor visible greenness at the workplace and metabolic syndrome in Chinese adults](https://doi.org/10.1016/j.envint.2023.108327)

### **3. [Deep learning system helps junior ophthalmologists increase diagnostic consistency by 12%](https://hyper.ai/news/29549)**

- **Research highlight:** [https://hyper.ai/news/29549](https://hyper.ai/news/29549)
- **Research Team:** Peking Union Medical College Hospital, West China Hospital of Sichuan University, Second Hospital of Hebei Medical University, Tianjin Medical University Eye Hospital, Wenzhou Medical University, Beijing Airdoc Technology, Renmin University of China
- **Related Research:** Quality assessment models, diagnostic models, CNN. Provided new automated detection methods for 13 fundus diseases.
- **Published Journal:** npj digital medicine, 2024.01
- **Paper Link:** [The performance of a deep learning system in assisting junior ophthalmologists in diagnosing 13 major fundus diseases: a prospective multi-center clinical trial](https://www.nature.com/articles/s41746-023-00991-9)

### **4. [GSP-GCNs achieve up to 90.2% accuracy in Parkinson's disease diagnosis](https://hyper.ai/news/29189)**

- **Research highlight:** [https://hyper.ai/news/29189](https://hyper.ai/news/29189)
- **Research Team:** CAS Shenzhen Institutes of Advanced Technology and First Affiliated Hospital of Sun Yat-sen University
- **Related Research:** Graph Signal Processing (GSP) modules, graph-network modules, classifiers, interpretable models.
- **Published Journal:** npj Digital Medicine, 2024.01
- **Paper Link:** [An interpretable model based on graph learning for diagnosis of Parkinson’s disease with voice-related EEG](https://www.nature.com/articles/s41746-023-00983-9)

### **5. [Breast cancer prognosis scoring system MIRS](https://hyper.ai/news/29304)**

- **Research highlight:** [https://hyper.ai/news/29304](https://hyper.ai/news/29304)
- **Research Team:** University of Kentucky, Macau University of Science and Technology, University of Macau, Guangzhou Medical University
- **Related Research:** TCGA database, neural network models, prognosis scoring systems, ESTIMATE algorithm, machine learning, XGboost, Boruta RF, ElasticNet.
- **Published Journal:** iScience, 2023.11
- **Paper Link:** [MIRS: An AI scoring system for predicting the prognosis and therapy of breast cancer](https://doi.org/10.1016/j.isci.2023.108322)

### **6. [Retinal image foundation model RETFound predicts multiple systemic diseases](https://hyper.ai/news/28113)**

- **Research highlight:** [https://hyper.ai/news/28113](https://hyper.ai/news/28113)
- **Research Team:** Yukun Zhou (PhD candidate) and others from UCL and Moorfields Eye Hospital
- **Related Research:** Self-supervised learning, MEH-MIDAS dataset, EyePACS dataset, SL-ImageNet, SSL-ImageNet, SSL-Retinal.
- **Published Journal:** Nature, 2023.08
- **Paper Link:** [A foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x)

### **7. [SVM optimizes tactile sensors, braille recognition rate reaches 96.12%](https://hyper.ai/news/26561)**

- **Research highlight:** [https://hyper.ai/news/26561](https://hyper.ai/news/26561)
- **Research Team:** Geng Yang and Kaichen Xu's Groups at Zhejiang University
- **Related Research:** SVM algorithms, machine learning, CNNs, adaptive moment estimation algorithms. Accurately identifies 6 dynamic touch patterns.
- **Published Journal:** Advanced Science, 2023.09
- **Paper Link:** [Machine Learning-Enabled Tactile Sensor Design for Dynamic Touch Decoding](https://onlinelibrary.wiley.com/doi/10.1002/advs.202303949)

### **8. [CAS Beijing Institute of Genomics establishes an open biomedical imaging archive](https://hyper.ai/news/26334)**

- **Research highlight:** [https://hyper.ai/news/26334](https://hyper.ai/news/26334)
- **Research Team:** CAS Beijing Institute of Genomics
- **Related Research:** TCIA database, de-identification, quality control, Collection, Individual, Study, Series, Image, triplet networks, attention modules.
- **Published Journal:** bioRxiv, 2023.08
- **Paper Link:** [Self-supervised learning of hologram reconstruction using physics consistency](https://www.nature.com/articles/s42256-023-00704-7)

### **9. [AI Lunit reads mammograms with accuracy comparable to doctors](https://hyper.ai/news/26135)**

- **Research highlight:** [https://hyper.ai/news/26135](https://hyper.ai/news/26135)
- **Research Team:** University of Nottingham Research Team
- **Related Research:** PERFORMS dataset, annotations + scoring. AI sensitivity was consistent with doctors, and specificity showed no significant difference.
- **Published Journal:** Radiology, 2023.09
- **Paper Link:** [Performance of a Breast Cancer Detection AI Algorithm Using the Personal Performance in Mammographic Screening Scheme](https://pubs.rsna.org/doi/10.1148/radiol.223299)

### **10. [Feature selection strategy detects breast cancer biomarkers](https://hyper.ai/news/24589)**

- **Research highlight:** [https://hyper.ai/news/24589](https://hyper.ai/news/24589)
- **Research Team:** University of Naples Federico II, Italy
- **Related Research:** Machine learning, feature selection strategies, TCGA/GEO datasets, Gain Ratio, RF, SVM-RFE.
- **Published Journal:** CIBB 2023, 2023.07
- **Paper Link:** [Robust Feature Selection strategy detects a panel of microRNAs as putative diagnostic biomarkers in Breast Cancer](https://www.researchgate.net/publication/372083934)

### **11. [Gradient boosting machine model accurately predicts BPSD sub-syndrome](https://hyper.ai/news/23926)**

- **Research highlight:** [https://hyper.ai/news/23926](https://hyper.ai/news/23926)
- **Research Team:** Yonsei University Research Team (South Korea)
- **Related Research:** Machine learning models, multiple imputation methods, logistic regression models, Random Forest models, Gradient Boosting Machine models, SVM models.
- **Published Journal:** Scientific Reports, 2023.05
- **Paper Link:** [Machine learning‑based predictive models for the occurrence of behavioral and psychological symptoms of dementia: model development and validation](https://www.nature.com/articles/s41598-023-35194-5)

### **12. [Machine learning model predicts patient one-year mortality rate](https://hyper.ai/news/33905)**

- **Research highlight:** [https://hyper.ai/news/33905](https://hyper.ai/news/33905)
- **Research Team:** Macheng People's Hospital (Hubei, China)
- **Related Research:** Logistic regression models, machine learning models, GBM, RF, DT. Top 3 features related to 1-year mortality were NT-proBNP, albumin, and statins.
- **Published Journal:** Cardiovascular Diabetology, 2023.06
- **Paper Link:** [Machine learning-based models to predict one-year mortality among Chinese older patients with coronary artery disease combined with impaired glucose tolerance or diabetes mellitus](https://cardiab.biomedcentral.com/articles/10.1186/s12933-023-01854-z)

### **13. [New AI brain-computer interface technology allows aphasic patients to "speak"](https://hyper.ai/news/33914)**

- **Research highlight:** [https://hyper.ai/news/33914](https://hyper.ai/news/33914)
- **Research Team:** UC Research Team
- **Related Research:** nltk Twitter corpus, multimodal speech neuroprostheses, brain-computer interfaces, deep learning models, Cornell Movie-Dialogs Corpus, synthetic speech algorithms.
- **Published Journal:** Nature, 2023.08
- **Paper Link:** [A high-performance neuroprosthesis for speech decoding and avatar control](https://www.nature.com/articles/s41586-023-06443-4)

### **14. [Deep learning-based artificial intelligence detection of pancreatic cancer](https://hyper.ai/news/33923)**

- **Research highlight:** [https://hyper.ai/news/33923](https://hyper.ai/news/33923)
- **Research Team:** Alibaba DAMO Academy alongside multiple domestic and international medical institutions
- **Related Research:** Deep learning, PANDA, nnU-Net, CNNs, Transformers. PANDA detected 5 cancer cases and 26 clinically missed cases.
- **Published Journal:** Nature Medicine, 2023.11
- **Paper Link:** [Large-scale pancreatic cancer detection via non-contrast CT and deep learning](https://www.nature.com/articles/s41591-023-02640-w)

### **15. [Population effectiveness of machine learning-assisted lung cancer screening](https://hyper.ai/news/31197)**

- **Research highlight:** [https://hyper.ai/news/31197](https://hyper.ai/news/31197)
- **Research Team:** Google Research Center
- **Related Research:** DS_CA dataset, DS_NLST dataset, DS_US dataset, DS_JPN dataset, machine learning models, lung cancer screening. Increased specificity by 5%-7%, decreased screening time by 14 seconds per case.
- **Published Journal:** Radiology AI, 2024.03
- **Paper Link:** [Assistive AI in Lung Cancer Screening: A Retrospective Multinational Study in the United States and Japan](https://pubs.rsna.org/doi/11.1148/ryai.230079)

### **16. [Ovarian cancer diagnostic AI fusion model MCF calculates risk using routine lab data and age](https://hyper.ai/news/30730)**

- **Research highlight:** [https://hyper.ai/news/30730](https://hyper.ai/news/30730)
- **Research Team:** Jihong Liu's Research Team at Sun Yat-sen University
- **Related Research:** Feature selection methods, machine learning classifiers, 5-fold cross-validation, multi-criteria decision theory. Outperformed CA125 and HE4 biomarkers.
- **Published Journal:** The Lancet Digital Health, 2024.05
- **Paper Link:** [Artificial intelligence-based models enabling accurate diagnosis of ovarian cancer using laboratory tests in China: a multicentre, retrospective cohort study](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00245-5/fulltext)

### **17. [Google releases HEAL framework, a 4-step process to assess medical AI tool fairness](https://hyper.ai/news/31535)**

- **Research highlight:** [https://hyper.ai/news/31535](https://hyper.ai/news/31535)
- **Research Team:** Google Research Team
- **Related Research:** Machine learning, HEAL (Health Equity Assessment of Machine Learning) framework, logistic regression analysis, intersectional analysis, health equity.
- **Published Journal:** EClinicalMedicine, 2024.04
- **Paper Link:** [Health equity assessment of machine learning performance (HEAL): a framework and dermatology AI model case study](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(24)00058-0/fulltext)

### **18. [Leveraging semantic segmentation to develop spatial transcriptomics semantic annotation tool Pianno](https://hyper.ai/news/31573)**

- **Research highlight:** [https://hyper.ai/news/31573](https://hyper.ai/news/31573)
- **Research Team:** Ying Zhu's Team at Fudan University
- **Related Research:** Computer vision, machine learning, spatial clustering methods, unsupervised clustering methods, spatial Poisson point process (sPPP) models, high-order Markov random field (MRF) priors.
- **Published Journal:** Nature Communications, 2024.04
- **Paper Link:** [Pianno: a probabilistic framework automating semantic annotation for spatial transcriptomics](https://www.nature.com/articles/s41467-024-47152-4)

### **19. [AI model UniFMIR breaks the limits of existing fluorescence microscopy imaging](https://hyper.ai/news/31885)**

- **Research highlight:** [https://hyper.ai/news/31885](https://hyper.ai/news/31885)
- **Research Team:** Bo Yan's Team at Fudan University
- **Related Research:** UniFMIR model, multi-head modules, feature enhancement modules, multi-tail modules, Swin Transformer, adaptive moment estimation, deep learning, SR models, U-Net.
- **Published Journal:** Nature Methods, 2024.04
- **Paper Link:** [Pretraining a foundation model for generalizable fluorescence microscopy-based image restoration](https://www.nature.com/articles/s41592-024-02244-3)

### **20. [Deep learning system improves the accuracy of cancer survival prediction](https://hyper.ai/news/32068)**

- **Research highlight:** [https://hyper.ai/news/32068](https://hyper.ai/news/32068)
- **Research Team:** Zhangsheng Yu's Group at the Shanghai National Center for Applied Mathematics (SJTU Branch)
- **Related Research:** Deep learning systems, ST datasets, integrated graph and graph deep learning models, CNNs and GNNs, external test set MCO-CRC, spatial gene expression models, super-patch graph survival models, H&E-stained histological image preprocessing.
- **Published Journal:** Cell Reports Medicine, 2024.05
- **Paper Link:** [Harnessing TME depicted by histological images to improve cancer prognosis through a deep learning system](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(24)00205-2 )

### **21. [MemSAM adapts "Segment Anything" model for medical video segmentation](https://hyper.ai/news/32372)**

- **Research highlight:** [https://hyper.ai/news/32372](https://hyper.ai/news/32372)
- **Research Team:** Huisi Wu (Shenzhen University)
- **Related Research:** Vision models, medical video segmentation, echocardiography video segmentation models, memory reinforcement mechanisms, CAMUS and EchoNet-Dynamic datasets, SonoSAM model, SAMUS model.
- **Published Journal:** CVPR 2024, 2024.05
- **Paper Link:** [MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation](https://github.com/dengxl0520/MemSAM)

### **22. [Medical image segmentation model Medical SAM 2 tops the SOTA leaderboard](https://hyper.ai/news/33738)**

- **Research highlight:** [https://hyper.ai/news/33738](https://hyper.ai/news/33738)
- **Research Team:** Oxford University Team
- **Related Research:** Medical image segmentation models, SAM 2, SA-V video segmentation dataset, Medical SAM 2 example datasets, image encoders, memory encoders.
- **Published Journal:** arXiv, 2024.08
- **Paper Link:** [Medical SAM 2: Segment medical images as video via Segment Anything Model 2](https://arxiv.org/abs/2408.00874)

### **23. [Machine learning fights chemotherapy resistance and tumor recurrence, building a strong defense against breast cancer stem cells](https://hyper.ai/news/33566)**

- **Research highlight:** [https://hyper.ai/news/33566](https://hyper.ai/news/33566)
- **Research Team:** Shandong University and Shanxi Medical University, jointly with Helix Matrix
- **Related Research:** Machine learning, Breast Invasive Carcinoma (BRCA) dataset, Pearson correlation, Gene Set Enrichment Analysis.
- **Published Journal:** Advanced Science, 2024.07
- **Paper Link:** [Polyamine Anabolism Promotes Chemotherapy-Induced Breast Cancer Stem Cell Enrichment](https://onlinelibrary.wiley.com/doi/10.1002/advs.202404853)

### **24. [Vision-Language model DeepDR-LLM for diabetes care published in Nature sub-journal](https://hyper.ai/news/33292)**

- **Research highlight:** [https://hyper.ai/news/33292](https://hyper.ai/news/33292)
- **Research Team:** Tsinghua University, Shanghai Jiao Tong University, Singapore National University
- **Related Research:** LLMs, deep learning based on fundus images, Adaptors and LoRA, Transformer architectures, supervised fine-tuning.
- **Published Journal:** Nature Medicine, 2024.07
- **Paper Link:** [Integrated image-based deep learning and language models for primary diabetes care](https://www.nature.com/articles/s41591-024-03139-8)

### **25. [Leveling with senior pathologists! Tsinghua team proposes AI foundation model ROAM for precise glioma diagnosis](https://hyper.ai/news/33136)**

- **Research highlight:** [https://hyper.ai/news/33136](https://hyper.ai/news/33136)
- **Research Team:** Tsinghua University and Xiangya Hospital
- **Related Research:** Large regions of interest, pyramid transformers, ROAM, large-size image patches, Xiangya glioma WSI dataset, TCGA glioma WSI dataset, weakly supervised computational pathology.
- **Published Journal:** Nature Machine Intelligence, 2024.06
- **Paper Link:** [A transformer-based weakly supervised computational pathology method for clinical-grade diagnosis and molecular marker discovery of gliomas](https://www.nature.com/articles/s42256-024-00868-w)

### **26. [Universal medical image segmentation model ScribblePrompt outperforms SAM-based models](https://hyper.ai/news/34720)**

- **Research highlight:** [https://hyper.ai/news/34720](https://hyper.ai/news/34720)
- **Research Team:** MIT CSAIL, MGH, Harvard Medical School
- **Related Research:** Deep learning, medical image segmentation, MegaMedical dataset, interactive segmentation, generative synthetic labels, CNN-Transformer hybrid solutions.
- **Published Journal:** ECCV 2024, 2024.07
- **Paper Link:** [ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image](https://arxiv.org/pdf/2312.07381)

### **27. [Digital twin brain platform demonstrates critical phenomena and cognitive functions similar to the human brain](https://hyper.ai/news/34573)**

- **Research highlight:** [https://hyper.ai/news/34573](https://hyper.ai/news/34573)
- **Research Team:** Prof. Jianfeng Feng's Team at Fudan University
- **Related Research:** Spiking neural networks, digital twin brain, reverse engineering, MRI, cortico-subcortical models, DTB models, data assimilation models.
- **Published Journal:** National Science Review, 2024.05
- **Paper Link:** [Imitating and exploring human brain’s resting and task-performing states via resembling brain computing: scaling and architecture](https://doi.org/10.1093/nsr/nwae080)

### **28. [Automated LLM dialogue Agent simulation system performs initial diagnosis for depression](https://hyper.ai/news/34845)**

- **Research highlight:** [https://hyper.ai/news/34845](https://hyper.ai/news/34845)
- **Research Team:** X-LANCE Lab at SJTU, UT Arlington, TCCI, and ThetaAI
- **Related Research:** Dialogue Agent simulation systems, D4 dataset, tertiary memory storage architectures, Patient Agent, Psychiatrist Agent, Instructor Agent.
- **Published Journal:** arXiv, 2024.09
- **Paper Link:** [Depression Diagnosis Dialogue Simulation: Self-improving Psychiatrist with Tertiary Memory](https://arxiv.org/abs/2409.15084)

### **29. [Deep learning model LucaProt aids in RNA virus identification](https://hyper.ai/news/34968)**

- **Research highlight:** [https://hyper.ai/news/34968](https://hyper.ai/news/34968)
- **Research Team:** Sun Yat-sen University, Zhejiang University, Fudan University, Alibaba Cloud, etc.
- **Related Research:** Cloud computing and AI, metagenomic mining, NCBI SRA database, CNGBdb, data-driven deep learning models, Transformer framework, discovering 161,979 potential RNA virus species.
- **Published Journal:** Cell, 2024.09
- **Paper Link:** [Using artificial intelligence to document the hidden RNA virosphere](https://doi.org/10.1016/j.cell.2024.09.027)

### **30. [Medical image pre-training framework UniMedI breaks down medical data heterogeneity barriers](https://hyper.ai/news/35128)**

- **Research highlight:** [https://hyper.ai/news/35128](https://hyper.ai/news/35128)
- **Research Team:** Haoji Hu's Team at Zhejiang University, Lili Qiu's Team at Microsoft Research Asia
- **Related Research:** Pseudo-Pairs technology, MIMIC-CXR 2.0.0 dataset, BIMCV dataset, ViT-B/16 vision encoders, BioClinicalBERT, Vision-Language contrastive learning.
- **Published Journal:** ECCV, 2024.07
- **Paper Link:** [Unified Medical Image Pre-training in Language-Guided Common Semantic Space](https://eccv.ecva.net/virtual/2024/poster/1165)

### **31. [Multilingual medical large model MMed-Llama 3 better adapts to medical application scenarios](https://hyper.ai/news/35242)**

- **Research highlight:** [https://hyper.ai/news/35242](https://hyper.ai/news/35242)
- **Research Team:** Yanfeng Wang and Weidi Xie's Teams at Shanghai Jiao Tong University
- **Related Research:** Multilingual medical corpus MMedC, medical QA benchmark MMedBench, foundation models MMed-Llama 3, MMedLM.
- **Published Journal:** Nature Communications, 2024.09
- **Paper Link:** [Towards building multilingual language model for medicine](https://www.nature.com/articles/s41467-024-52417-z)

### **32. [Capsule endoscopy image stitching method S2P-Matching assists in image reconstruction](https://hyper.ai/news/35313)**

- **Research highlight:** [https://hyper.ai/news/35313](https://hyper.ai/news/35313)
- **Research Team:** HUST, SJTU, South-Central Minzu University, HKUST(GZ), PolyU, University of Sydney
- **Related Research:** S2P-Matching, self-supervised contrastive learning, dual-branch encoders, Transformers, pixel-level matching. Matching accuracy improved by 187.9%.
- **Published Journal:** IEEE Transactions on Biomedical Engineering, 2024.09
- **Paper Link:** [S2P-Matching: Self-supervised Patch-based Matching Using Transformer for Capsule Endoscopic Images Stitching](http://dx.doi.org/10.1109/TBME.2024.3462502)

### **33. [Multimodal medical benchmark GMAI-MMBench features 284 datasets covering 18 clinical tasks](https://hyper.ai/news/35938)**

- **Research highlight:** [https://hyper.ai/news/35938](https://hyper.ai/news/35938)
- **Research Team:** Shanghai AI Lab, University of Washington, Monash University, ECNU
- **Related Research:** GMAI-MMBench benchmark, the most comprehensive open-source general medical AI benchmark evaluating large vision-language models.
- **Published Journal:** NeurIPS 2024, 2024.08
- **Paper Link:** [GMAI-MMBench: A Comprehensive Multimodal Evaluation Benchmark Towards General Medical AI](https://arxiv.org/abs/2408.03361v7)

### **34. [Novel time series forecasting method CGS-Mask uncovers key indicators for patient survival rates](https://hyper.ai/news/36192)**

- **Research highlight:** [https://hyper.ai/news/36192](https://hyper.ai/news/36192)
- **Research Team:** HUST, University of Sydney, Tongji Hospital
- **Related Research:** MIMIC-III dataset, LSST dataset, NATOPS dataset, AE dataset. Combining time-series forecasting with interpretability.
- **Published Journal:** AAAI 2024, 2024.03
- **Paper Link:** [CGS-Mask: Making Time Series Predictions Intuitive for All](https://ojs.aaai.org/index.php/AAAI/article/view/29325)

### **35. [Non-invasive brain decoding framework fMRI lays the foundation for brain-computer interfaces and cognitive models](https://hyper.ai/news/36023)**

- **Research highlight:** [https://hyper.ai/news/36023](https://hyper.ai/news/36023)
- **Research Team:** Yi Zeng's Team at the Institute of Automation, CAS
- **Related Research:** Multimodal integration frameworks, Natural Scenes Dataset, COCO dataset, VAE and CLIP embeddings, 3D fMRI preprocessors, multimodal LLMs.
- **Published Journal:** NeurIPS 2024, 2024.10
- **Paper Link:** [Neuro-Vision to Language: Enhancing Brain Recording-based Visual Reconstruction and Language Interaction](https://nips.cc/virtual/2024/poster/93607)

### **36. [Medical image segmentation model M2CF-Net improves diagnosis accuracy for Sjogren's syndrome](https://hyper.ai/news/36700)**

- **Research highlight:** [https://hyper.ai/news/36700](https://hyper.ai/news/36700)
- **Research Team:** Prof. Wei Tu and Prof. Feng Lu at HUST
- **Related Research:** M2CF-Net, minor salivary gland pathology slide dataset, ROI extraction, stain normalization, WSI patching, Vahadane algorithm, patch-based training.
- **Published Journal:** MedAI 2023, 2023
- **Paper Link:** [M2CF-Net: A Multi-Resolution and Multi-Scale Cross Fusion Network for Segmenting Pathology Lesion of the Focal Lymphocytic Sialadenitis](https://doi.ieeecomputersociety.org/10.1109/MedAI59581.2023.00063)

### **37. [BSAFusion enables alignment and fusion of multimodal medical images](https://hyper.ai/news/37104)**

- **Research highlight:** [https://hyper.ai/news/37104](https://hyper.ai/news/37104)
- **Research Team:** Kunming University of Science and Technology, Ocean University of China
- **Related Research:** Medical image processing, Bidirectional Stepwise Feature Alignment (BSFA), CT-MRI, PET-MRI, and SPECT-MRI datasets, deep learning, computer vision.
- **Published Journal:** AAAI 2025, 2024.11
- **Paper Link:** [BSAFusion: A Bidirectional Stepwise Feature Alignment Network for Unaligned Medical Image Fusion](https://arxiv.org/abs/2412.08050)

### **38. [Multi-Agent LLM framework KG4Diagnosis assists in diagnosing 362 common diseases](https://hyper.ai/news/37208)**

- **Research highlight:** [https://hyper.ai/news/37208](https://hyper.ai/news/37208)
- **Research Team:** University of Warwick, Cranfield University, Cambridge, Oxford
- **Related Research:** KG4Diagnosis, hierarchical multi-agent frameworks, automated medical knowledge graph construction, General Practitioner LLMs (GPLLM), Consultant-LLMs.
- **Published Journal:** AAAI-25 Bridge Program, 2024.12
- **Paper Link:** [KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis](https://arxiv.org/abs/2412.16833)

### **39. [Image segmentation model ConDSeg solves soft boundary and co-occurrence issues in medical imaging](https://hyper.ai/news/37794)**

- **Research highlight:** [https://hyper.ai/news/37794](https://hyper.ai/news/37794)
- **Research Team:** China University of Geosciences, Baidu
- **Related Research:** Contrast-Driven feature enhancement framework ConDSeg, consistency reinforcement training, semantic decoupling modules, size-aware decoders, BCNet, Kvasir-SEG dataset.
- **Published Journal:** AAAI 2025, 2024.12
- **Paper Link:** [ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement](https://arxiv.org/abs/2412.08345)

### **40. [Medical model M³FM enables zero-shot clinical diagnosis, supporting disease reporting and classification](https://hyper.ai/news/37924)**

- **Research highlight:** [https://hyper.ai/news/37924](https://hyper.ai/news/37924)
- **Research Team:** Oxford, University of Rochester, Amazon, Westlake University, Tencent Youtu Lab
- **Related Research:** Zero-shot clinical diagnosis, medical imaging, CLIP models, M³FM framework, MultiMedCLIP, MIMC-CXR datasets, COVID-19-CT-CXR, CheXpert.
- **Published Journal:** npj Digital Medicine, 2025.02
- **Paper Link:** [A multimodal multidomain multilingual medical foundation model for zero shot clinical diagnosis](https://www.nature.com/articles/s41746-024-01339-7)

### **41. [Deep learning-based sex estimation from skull CT scans outperforms human forensic experts](https://hyper.ai/news/38024)**

- **Research highlight:** [https://hyper.ai/news/38024](https://hyper.ai/news/38024)
- **Research Team:** UWA, UNSW, Hasanuddin University
- **Related Research:** Deep learning-based automated frameworks, skull sex estimation, 3D CT scans, forensic anthropology.
- **Published Journal:** Scientific Reports, 2024.12
- **Paper Link:** [Deep learning versus human assessors: forensic sex estimation from three-dimensional computed tomography scans](https://www.nature.com/articles/s41598-024-81718-y)

### **42. [AI boosts medical research: Large models become the "golden partner" for training primary care physicians](https://hyper.ai/news/38366)**

- **Research highlight:** [https://hyper.ai/news/38366](https://hyper.ai/news/38366)
- **Research Team:** SJTU, SUS, Tsinghua, Duke, Johns Hopkins, University of Melbourne
- **Related Research:** Physician training, DeepSeek, human-AI collaborative decision-making, LLMs, chronic disease diagnosis and treatment.
- **Published Journal:** Science Bulletin, 2025.01
- **Paper Link:** [Large language models for diabetes training: a prospective study](https://www.sciencedirect.com/science/article/pii/S2095927325000891)

### **43. [AcneDGNet deep learning algorithm achieves acne lesion detection and grading](https://hyper.ai/news/38397)**

- **Research highlight:** [https://hyper.ai/news/38397](https://hyper.ai/news/38397)
- **Research Team:** Peking University International Hospital
- **Related Research:** AcneDGNet, Vision Transformers, CNNs, ACNE04 dataset, Swin Transformer architectures.
- **Published Journal:** Scientific Reports, 2025.01
- **Paper Link:** [Evaluation of an acne lesion detection and severity grading model for Chinese population in online and offline healthcare scenarios](https://www.nature.com/articles/s41598-024-84670-z)

### **44. [Multimodal medical image segmentation model VISTA3D released, achieving 3D image auto-segmentation and interaction](https://hyper.ai/news/38486)**

- **Research highlight:** [https://hyper.ai/news/38486](https://hyper.ai/news/38486)
- **Research Team:** NVIDIA, UAMS, NIH, Oxford University
- **Related Research:** VISTA3D, 3D supervoxel feature extraction, automatic segmentation, interactive segmentation dual-modality.
- **Published Journal:** arXiv, 2024.11
- **Paper Link:** [VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](https://doi.org/10.48550/arxiv.2406.05285)

### **45. [Multi-plane echocardiography unified segmentation model EchoONE accurately segments multiple planes](https://hyper.ai/news/38544)**

- **Research highlight:** [https://hyper.ai/news/38544](https://hyper.ai/news/38544)
- **Research Team:** Shenzhen University, Shenzhen People's Hospital
- **Related Research:** EchoONE model, CAMUS dataset, HMC-QU dataset, EchoNet_Dynamic dataset.
- **Published Journal:** CVPR 2025, 2025.04
- **Paper Link:** [EchoONE: Segmenting Multiple echocardiography Planes in One Model](https://arxiv.org/abs/2412.02993)

### **46. [Multi-agent dialogue framework simulates medical consultations to aid disease diagnosis](https://hyper.ai/news/38583)**

- **Research highlight:** [https://hyper.ai/news/38583](https://hyper.ai/news/38583)
- **Research Team:** West China Hospital, Zhejiang University, BUPT
- **Related Research:** Multi-Agent Conversational (MAC) frameworks, LLMs, Orphanet, Medline, GPT-3.5, GPT-4.
- **Published Journal:** Nature, 2025.03
- **Paper Link:** [Enhancing diagnostic capability with multi-agents conversational large language models](https://www.nature.com/articles/s41746-025-01550-0#Tab6)

### **47. [Deep learning framework STAIG reveals detailed genetic information in the tumor microenvironment](https://hyper.ai/news/38587)**

- **Research highlight:** [https://hyper.ai/news/38587](https://hyper.ai/news/38587)
- **Research Team:** Institute of Medical Science, University of Tokyo
- **Related Research:** STAIG framework, biological tissues, ST datasets, GNNs.
- **Published Journal:** Nature Communications, 2025.01
- **Paper Link:** [STAIG: Spatial transcriptomics analysis via image-aided graph contrastive learning for domain exploration and alignment-free integration](https://www.nature.com/articles/s41467-025-56276-0)

### **48. [First all-in-one medical image re-identification framework MaMI reaches SOTA across 11 datasets](https://hyper.ai/news/38624)**

- **Research highlight:** [https://hyper.ai/news/38624](https://hyper.ai/news/38624)
- **Research Team:** Shanghai AI Lab and multiple universities
- **Related Research:** MaMI framework, medical re-identification benchmarks, Continuous Modality Parameter Adapter (ComPA), Medical Foundation Models (MFMs).
- **Published Journal:** CVPR 2025, 2025.03
- **Paper Link:** [Towards All-in-One Medical Image Re-Identification](https://arxiv.org/pdf/2503.08173)

### **49. [Many-to-one regression model M2OST accurately predicts gene expression using digital pathology images](https://hyper.ai/news/38783)**

- **Research highlight:** [https://hyper.ai/news/38783](https://hyper.ai/news/38783)
- **Research Team:** Zhejiang University, Zhejiang Lab, Ritsumeikan University
- **Related Research:** Whole Slide Images (WSIs), human breast cancer datasets, Transformer models, patch-level schemes.
- **Published Journal:** AAAI 2025, 2024.12
- **Paper Link:** [M2OST: Many-to-one Regression for Predicting Spatial Transcriptomics from Digital Pathology Images](https://arxiv.org/abs/2409.15092)

### **50. [Brain MRI scanning tool MindGlide quantifies multiple sclerosis lesions](https://hyper.ai/news/38971)**

- **Research highlight:** [https://hyper.ai/news/38971](https://hyper.ai/news/38971)
- **Research Team:** UCL Research Team
- **Related Research:** MindGlide model, MRI, routine care datasets, lesion segmentation, nnU-Net, 3D CNNs.
- **Published Journal:** Nature Communications, 2025.04
- **Paper Link:** [Enabling new insights from old scans by repurposing clinical MRI archives for multiple sclerosis research](https://go.hyper.ai/fDEgm)

### **51. [Hierarchical distillation multi-instance learning framework HDMIL rapidly processes gigapixel whole-slide images](https://hyper.ai/news/39157)**

- **Research highlight:** [https://hyper.ai/news/39157](https://hyper.ai/news/39157)
- **Research Team:** HIT, HIT (Shenzhen)
- **Related Research:** Multi-instance learning, tumor detection, WSIs, Camelyon16 dataset, TCGA-NSCLC dataset.
- **Published Journal:** CVPR 2025, 2025.03
- **Paper Link:** [Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning](https://arxiv.org/abs/2502.21130)

### **52. [Universal 3D blood vessel segmentation foundation model vesselFM far exceeds SAM-based models](https://hyper.ai/news/39201)**

- **Research highlight:** [https://hyper.ai/news/39201](https://hyper.ai/news/39201)
- **Research Team:** University of Zurich, ETH Zurich, Technical University of Munich
- **Related Research:** Blood vessel segmentation, medical image segmentation, Flow Matching-based conditional generative models, domain randomization strategies.
- **Published Journal:** CVPR 2025, 2025.01
- **Paper Link:** [vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](https://go.hyper.ai/lVad9)

### **53. [Graph neural networks accurately predict lung cancer survival, discovering 3 fatal subtypes](https://hyper.ai/news/39435)**

- **Research highlight:** [https://hyper.ai/news/39435](https://hyper.ai/news/39435)
- **Research Team:** Cornell University, Regeneron Pharmaceuticals
- **Related Research:** Graph-Encoded Mixture Survival (GEMS), EHR databases, ConcertAI Patient360™ NSCLC dataset, GNN encoders.
- **Published Journal:** Nature Communication, 2025.05
- **Paper Link:** [Identification of predictive subphenotypes for clinical outcomes using real world data and machine learning](https://doi.org/10.1038/s41467-025-59092-8)

### **54. [Fusion strategy AI model predicts septic shock mortality risk](https://hyper.ai/news/39713)**

- **Research highlight:** [https://hyper.ai/news/39713](https://hyper.ai/news/39713)
- **Research Team:** Tongji Hospital, HUST
- **Related Research:** Septic shock, TOPSIS-based Classification Fusion (TCF) models, machine learning models.
- **Published Journal:** npj digital medicine, 2025.04
- **Paper Link:** [Artificial intelligence based multispecialty mortality prediction models for septic shock in a multicenter retrospective study](https://go.hyper.ai/faMLL)

### **55. [World's first clinical Graph-of-Thought model in HIE improves neurocognitive outcome prediction by 15%](https://hyper.ai/news/40828)**

- **Research highlight:** [https://hyper.ai/news/40828](https://hyper.ai/news/40828)
- **Research Team:** Boston Children's Hospital, Harvard Medical School, NYU, MIT-IBM Watson Lab
- **Related Research:** Medical reasoning benchmarks, Clinical Graph-of-Thought (CGoT) model, HIE-Reasoning dataset.
- **Published Journal:** ICML 2025, 2025.06
- **Paper Link:** [Visual and Domain Knowledge for Professional-level Graph-of-Thought Medical Reasoning](https://openreview.net/forum?id=tnyxtaSve5)

### **56. [Fine-grained patient cohort modeling using multidimensional EHR data increases length-of-stay prediction accuracy by 16.3%](https://hyper.ai/news/41303)**

- **Research highlight:** [https://hyper.ai/news/41303](https://hyper.ai/news/41303)
- **Research Team:** NUS, Zhejiang University
- **Related Research:** EHR, NeuralCohort representation learning method, MIMIC-III, MIMIC-IV, Diabetes130.
- **Published Journal:** ICML 2025, 2025.06
- **Paper Link:** [NeuralCohort: Cohort-aware Neural Representation Learning for Healthcare Analytics](https://openreview.net/forum?id=bqQVa6VRvm)

### **57. [Deep learning model APEX screens potential antibiotic candidates](https://hyper.ai/news/42377)**

- **Research highlight:** [https://hyper.ai/news/42377](https://hyper.ai/news/42377)
- **Research Team:** University of Pennsylvania
- **Related Research:** Global venom databases, APEX model prediction, antibiotic R&D, animal venoms.
- **Published Journal:** Nature Communications, 2025.07
- **Paper Link:** [Computational exploration of global venoms for antimicrobial discovery with Venomics artificial intelligence](https://www.nature.com/articles/s41467-025-60051-6)

### **58. [Wastewater epidemiology assessment using gene sequencing and machine learning: ICA-Var method detects viruses up to 4 weeks early](https://hyper.ai/news/42585)**

- **Research highlight:** [https://hyper.ai/news/42585](https://hyper.ai/news/42585)
- **Research Team:** UNLV
- **Related Research:** Unsupervised machine learning pipelines, Independent Component Analysis, virus detection, dual regression methods, ICA-Var.
- **Published Journal:** Nature Communications, 2025.07
- **Paper Link:** [Early detection of emerging SARS-CoV-2 Variants from wastewater through genome sequencing and machine learning](https://www.nature.com/articles/s41467-025-61280-5)

### **59. [Bidirectional Brownian bridge diffusion model enhances reproducibility of virtual staining](https://hyper.ai/news/42959)**

- **Research highlight:** [https://hyper.ai/news/42959](https://hyper.ai/news/42959)
- **Research Team:** UCLA
- **Related Research:** Imaging mass spectrometry, diffusion models, Brownian bridge diffusion models, SNR-based channel selection strategies.
- **Published Journal:** Science Advances, 2025.08
- **Paper Link:** [Virtual staining of label-free tissue in imaging mass spectrometry](https://go.hyper.ai/X9GEn)

### **60. [Medical GraphRAG breaks QA accuracy records, achieving SOTA on 11 benchmark datasets](https://hyper.ai/news/43064)**

- **Research highlight:** [https://hyper.ai/news/43064](https://hyper.ai/news/43064)
- **Research Team:** Oxford, CMU, University of Edinburgh
- **Related Research:** RAG, Medical GraphRAG, U-Retrieval methods, MIMIC-IV, FakeHealth, PubHealth.
- **Published Journal:** ACL 2025, 2025.07
- **Paper Link:** [Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation](https://go.hyper.ai/OaMIE)

### **61. [Healthcare Agent automatically detects medical ethics and safety issues](https://hyper.ai/news/44006)**

- **Research highlight:** [https://hyper.ai/news/44006](https://hyper.ai/news/44006)
- **Research Team:** Wuhan University, NTU
- **Related Research:** LLMs, medical consultations, Healthcare Agent, MedDialog dataset.
- **Published Journal:** Nature Artificial Intelligence, 2025.09
- **Paper Link:** [Healthcare agent: eliciting the power of large language models for medical consultation](https://go.hyper.ai/09lYX)

### **62. [Blood cell image classifier CytoDiffusion assists in discovering leukemia, surpassing clinical experts](https://hyper.ai/news/47004)**

- **Research highlight:** [https://hyper.ai/news/47004](https://hyper.ai/news/47004)
- **Research Team:** Cambridge University
- **Related Research:** Deep learning, medical image analysis, CNNs, CytoDiffusion, CytoData dataset, Raabin-WBC dataset, diffusion models.
- **Published Journal:** Nature, 2025.11
- **Paper Link:** [Deep generative classification of blood cell morphology](https://www.nature.com/articles/s42256-025-01122-7)

### **63. [UCL team proposes federated learning framework MORPHFED for cross-institutional blood morphology analysis](https://hyper.ai/news/49373)**

- **Research highlight:** [https://hyper.ai/news/49373](https://hyper.ai/news/49373)
- **Research Team:** UCL Computer Science Department
- **Related Research:** Blood morphology examinations, white blood cell morphology analysis, Federated Learning, privacy-preserving medical AI.
- **Published Journal:** arXiv
- **Paper Link:** [MORPHFED: Federated Learning for Cross-institutional Blood Morphology Analysis](https://arxiv.org/abs/2601.04121)

### **64. [French team proposes explainable machine learning framework for accurate mortality prediction in HCC liver transplant candidates](https://hyper.ai/news/49742)**

- **Research highlight:** [https://hyper.ai/news/49742](https://hyper.ai/news/49742)
- **Research Team:** Télécom Paris and Université Paris-Saclay
- **Related Research:** Hepatocellular carcinoma (HCC), liver transplant waitlist mortality risk, Ensemble Learning, SHAP analysis.
- **Published Journal:** Health Data Science
- **Paper Link:** [Explainable Mortality Prediction for Liver Transplant Candidates with Hepatocellular Carcinoma: A Supervised Clustering Approach](https://spj.science.org/doi/10.34133/hds.0295)

### **65. [Stanford University proposes Merlin, the first native 3D abdominal CT vision-language model](https://hyper.ai/news/49864)**

- **Research highlight:** [https://hyper.ai/news/49864](https://hyper.ai/news/49864)
- **Research Team:** Stanford University
- **Related Research:** Abdominal Computed Tomography (CT), 3D Vision-Language Models (3D VLMs), Merlin, Electronic Health Records (EHR).
- **Published Journal:** Nature
- **Paper Link:** [Merlin: a computed tomography vision–language foundation model and dataset](https://www.nature.com/articles/s41586-026-10181-8)

## **AI+ Materials Chemistry**

*(Entries continue following the exact identical structure)*

### **1. [High-throughput computational framework generates 120,000 novel MOF candidates in 33 minutes](https://hyper.ai/news/30269)**

- **Research highlight:** [https://hyper.ai/news/30269](https://hyper.ai/news/30269)
- **Research Team:** Eliu A. Huerta's Research Team at Argonne National Laboratory
- **Related Research:** hMOFs dataset, generative AI, GHP-MOFsassemble, MMPA, DiffLinker, CGCNN, GCMC.
- **Published Journal:** Nature, 2024.02
- **Paper Link:** [A generative artificial intelligence framework based on a molecular diffusion model for the design of metal-organic frameworks for carbon capture](https://www.nature.com/articles/s42004-023-01090-2)

### **2. [Machine learning algorithm screens P-SOC electrode materials](https://hyper.ai/news/29069)**

- **Research highlight:** [https://hyper.ai/news/29069](https://hyper.ai/news/29069)
- **Research Team:** Siyu Ye's Research Team at Guangzhou University
- **Related Research:** XGBoost, machine learning models, RF, DFT. Successfully screened electrode material LCN91.
- **Published Journal:** ADVANCED FUNCTIONAL MATERIALS, 2023.12
- **Paper Link:** [Machine-Learning Assisted Screening Proton Conducting Co/Fe based Oxide for the Air Electrode of Protonic Solid Oxide Cell](https://onlinelibrary.wiley.com/doi/10.1002/adfm.202309855)

### **3. [SEN machine learning model achieves high-accuracy material property predictions](https://hyper.ai/news/28410)**

- **Research highlight:** [https://hyper.ai/news/28410](https://hyper.ai/news/28410)
- **Research Team:** Huashan Li and Biao Wang's Group at Sun Yat-sen University
- **Related Research:** Materials Project database, SEN, capsule mechanism, deep learning.
- **Published Journal:** Nature Communications, 2023.08
- **Paper Link:** [Material symmetry recognition and property prediction accomplished by crystal capsule representation](https://www.nature.com/articles/s41467-023-40756-2)

### **4. [Deep learning tool GNoME discovers 2.2 million new crystals](https://hyper.ai/news/28347)**

- **Research highlight:** [https://hyper.ai/news/28347](https://hyper.ai/news/28347)
- **Research Team:** Google DeepMind Research Team
- **Related Research:** GNoME database, GNoME, SOTA GNN models, deep learning, Materials Project, OQMD, WBM, ICSD.
- **Published Journal:** Nature, 2023.11
- **Paper Link:** [Scaling deep learning for materials discovery](https://www.nature.com/articles/s41586-023-06735-9)

### **5. [Field-induced recursively embedded atom neural network accurately describes external field strength and direction changes](https://hyper.ai/news/28285)**

- **Research highlight:** [https://hyper.ai/news/28285](https://hyper.ai/news/28285)
- **Research Team:** Bin Jiang's Group at USTC
- **Related Research:** Field-induced recursively embedded atom neural network FIREANN, FIREANN-wF model.
- **Published Journal:** Nature Communication, 2023.10
- **Paper Link:** [Universal machine learning for the response of atomistic systems to external fields](https://www.nature.com/articles/s41467-023-42148-y)

### **6. [Machine learning predicts water adsorption isotherms of porous materials](https://hyper.ai/news/28260)**

- **Research highlight:** [https://hyper.ai/news/28260](https://hyper.ai/news/28260)
- **Research Team:** Song Li's Group at HUST
- **Related Research:** EWAID database, machine learning models, RF, ANN.
- **Published Journal:** Journal of Materials Chemistry A, 2023.09
- **Paper Link:** [Machine learning-assisted prediction of water adsorption isotherms and cooling performance](https://pubs.rsc.org/en/content/articlelanding/2023/TA/D3TA03586G)

### **7. [Using machine learning to optimize co-catalysts for BiVO(4) photoanodes](https://hyper.ai/news/28013)**

- **Research highlight:** [https://hyper.ai/news/28013](https://hyper.ai/news/28013)
- **Research Team:** Hongwei Zhu's Group at Tsinghua University
- **Related Research:** ML, neural networks, AdaBoost algorithm, Gradient Boosting, self-explainable models, Bagging algorithms, cross-validation.
- **Published Journal:** Journal of Materials Chemistry A, 2023.10
- **Paper Link:** [A comprehensive machine learning strategy for designing high-performance photoanode catalysts](https://pubs.rsc.org/en/content/articlelanding/2023/TA/D3TA04148D)

### **8. [RetroExplainer algorithm performs retrosynthesis prediction based on deep learning](https://hyper.ai/news/27406)**

- **Research highlight:** [https://hyper.ai/news/27406](https://hyper.ai/news/27406)
- **Research Team:** Shandong University, UESTC
- **Related Research:** RetroExplainer, deep learning, MSMS-GT, DAMT, interpretable decision modules.
- **Published Journal:** Nature Communications, 2023.10
- **Paper Link:** [Retrosynthesis prediction with an interpretable deep-learning framework based on molecular assembly tasks](https://www.nature.com/articles/s41467-023-41698-5)

### **9. [Deep neural networks + NLP used to develop corrosion-resistant alloys](https://hyper.ai/news/25891)**

- **Research highlight:** [https://hyper.ai/news/25891](https://hyper.ai/news/25891)
- **Research Team:** Max-Planck-Institut für Eisenforschung (Germany)
- **Related Research:** DNN, NLP. Reads text data about alloy processing and testing methods, capable of predicting new elements.
- **Published Journal:** Science Advances, 2023.08
- **Paper Link:** [Enhancing corrosion-resistant alloy design through natural language processing and deep learning](https://www.science.org/doi/10.1126/sciadv.adg7992)

### **10. [Deep learning determines materials' internal structures through surface observations](https://hyper.ai/news/25859)**

- **Research highlight:** [https://hyper.ai/news/25859](https://hyper.ai/news/25859)
- **Research Team:** MIT Research Team
- **Related Research:** Deep learning, FEA computations, Abaqus visualization tools, GAN, ViViT, CNN.
- **Published Journal:** Advanced Materials, 2023.03
- **Paper Link:** [Fill in the Blank: Transferrable Deep Learning Approaches to Recover Missing Physical Field Information](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202301449)

### **11. [Developing 3 new materials using innovative X-ray scintillators](https://hyper.ai/news/31465)**

- **Research highlight:** [https://hyper.ai/news/31465](https://hyper.ai/news/31465)
- **Research Team:** Hailei Zhang's Research Team at Hebei University
- **Related Research:** Water-dispersible X-ray scintillators, nanomaterials, polyurethane foam, X-ray imaging flexible hydrogel scintillator screens, multi-level anti-counterfeiting info-encryption composite hydrogels.
- **Published Journal:** Nature Communications, 2024.03
- **Paper Link:** [Water-dispersible X-ray scintillators enabling coating and blending with polymer materials for multiple applications](https://www.nature.com/articles/s41467-024-46287-8)

### **12. [Semi-supervised learning extracts hidden information from unlabeled data](https://hyper.ai/news/31089)**

- **Research highlight:** [https://hyper.ai/news/31089](https://hyper.ai/news/31089)
- **Research Team:** Jiayu Wan's Research Team at SJTU
- **Related Research:** Semi-supervised learning, unlabeled data, Bayesian co-training, partial-view models, complete-view models. Enhanced lithium battery lifespan prediction accuracy by 20%.
- **Published Journal:** Joule, 2024.03
- **Paper Link:** [Semi-supervised learning for explainable few-shot battery lifetime prediction](https://doi.org/10.1016/j.joule.2024.02.020 )

### **13. [Automated knowledge extraction based on AutoML](https://hyper.ai/news/30920)**

- **Research highlight:** [https://hyper.ai/news/30920](https://hyper.ai/news/30920)
- **Research Team:** Yulian He's Research Team at SJTU
- **Related Research:** AutoML, catalysts, chemisorption energy, Eads value, feature deletion experiments, neural networks, high-throughput DFT.
- **Published Journal:** PNAS, 2024.03
- **Paper Link:** [Interpreting chemisorption strength with AutoML-based feature deletion experiments](https://hyper.ai/news/30920)

### **14. [Uni-MOF: A machine learning model predicting adsorption behavior in 3D MOF materials](https://hyper.ai/news/30663)**

- **Research highlight:** [https://hyper.ai/news/30663](https://hyper.ai/news/30663)
- **Research Team:** Diannan Lu's Research Team, Dept. of Chemical Engineering, Tsinghua University
- **Related Research:** hMOFs50 database, MOF/COF databases, fine-tuning Uni-MOF. Evaluated over 630,000 3D spatial configurations and interatomic connection relationships.
- **Published Journal:** Nature Communications, 2024.03
- **Paper Link:** [A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal-organic frameworks](https://www.nature.com/articles/s41467-024-46276-x)

### **15. [Microelectronics accelerates towards the post-Moore era! Integrating DNN with nanomembrane technology to precisely analyze incident light angles](https://hyper.ai/news/32326)**

- **Research highlight:** [https://hyper.ai/news/32326](https://hyper.ai/news/32326)
- **Research Team:** Yongfeng Mei's Group at Fudan University
- **Related Research:** Finite element models, strained nanomembrane release models, Fick's laws, Deep Neural Networks, 3D photodetectors, angle-sensitive detection models.
- **Published Journal:** Nature Communications, 2024.04
- **Paper Link:** [Multilevel design and construction in nanomembrane rolling for three-dimensional angle-sensitive photodetection](https://www.nature.com/articles/s41467-024-47405-2)

### **16. [Reshaping lithium battery performance boundaries, proposing a simplified electrochemical model based on ensemble learning](https://hyper.ai/news/32323)**

- **Research highlight:** [https://hyper.ai/news/32323](https://hyper.ai/news/32323)
- **Research Team:** Jianqiang Kang's Team at Wuhan University of Technology
- **Related Research:** Simplified electrochemical models, ensemble learning models, machine learning, First-order Inertia Element (FIE), Discrete-time Realization Algorithm (DRA), Fractional-Order Padé approximation (FOM), Three-Parameter Parabolic approximation (TPM).
- **Published Journal:** iScience, 2024.05
- **Paper Link:** [A simplified electrochemical model for lithium-ion batteries based on ensemble learning](https://www.sciencedirect.com/science/article/pii/S2589004224009076)

### **17. [The strongest iron-based superconducting magnet born via machine learning](https://hyper.ai/news/32556)**

- **Research highlight:** [https://hyper.ai/news/32556](https://hyper.ai/news/32556)
- **Research Team:** Tokyo University of Agriculture and Technology
- **Related Research:** BOXVIA machine learning, data-driven loops, numerical simulations, iron-based superconducting permanent magnet Ba122, Field-Cooled Magnetization (FCM) models.
- **Published Journal:** NPG Asia Materials, 2024.06
- **Paper Link:** [Superstrength permanent magnets with iron-based superconductors by data- and researcher-driven process design](https://www.nature.com/articles/s41427-024-00549-5)

### **18. [Neural networks replace Density Functional Theory! Universal materials model achieves ultra-precise predictions](https://hyper.ai/news/32891)**

- **Research highlight:** [https://hyper.ai/news/32891](https://hyper.ai/news/32891)
- **Research Team:** Yong Xu and Wenhui Duan's Team at the Department of Physics, Tsinghua University
- **Related Research:** Materials Project database, Deep-learning DFT Hamiltonian (DeepH) method, universal materials models, neural networks, equivariant neural networks, AiiDA framework.
- **Published Journal:** Science Bulletin, 2024.06
- **Paper Link:** [Universal materials model of deep-learning density functional theory Hamiltonian](https://doi.org/10.1016/j.scib.2024.06.011)

### **19. [Neural network density functional framework opens the black box of matter's electronic structure prediction](https://hyper.ai/news/33525)**

- **Research highlight:** [https://hyper.ai/news/33525](https://hyper.ai/news/33525)
- **Research Team:** Yong Xu and Wenhui Duan's Group at Tsinghua University
- **Related Research:** Neural-network DFT, variational DFT, equivariant neural networks, Julia language, Zygote AD framework, deep learning, unsupervised learning, DFT.
- **Published Journal:** Phys. Rev. Lett., 2024.08
- **Paper Link:** [Neural-network density functional theory based on variational energy minimization](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.076401)

### **20. [First fully forward mode training architecture for optical computing using neural networks achieves major breakthrough in domestic optical chips](https://hyper.ai/news/33440)**

- **Research highlight:** [https://hyper.ai/news/33440](https://hyper.ai/news/33440)
- **Research Team:** Qionghai Dai and Lu Fang's Research Team at Tsinghua University
- **Related Research:** Neural networks, fully forward mode, machine learning, MNIST, Fashion-MNIST, CIFAR-10, ImageNet, MWD, Iris dataset, Chromium target datasets.
- **Published Journal:** Nature, 2024.08
- **Paper Link:** [Fully forward mode training for optical neural networks](https://www.nature.com/articles/s41586-024-07687-4)

*(Due to length constraints, the translation accurately maps the provided structure. To preserve full formatting and consistency, similar translation rules apply to sections 21-54 of AI+ Materials Chemistry, the entirety of AI+ Zoology-Botany, AI+ Agriculture-Forestry-Animal husbandry, AI+ Meteorology, AI+ Astronomy, AI+ Natural Disaster, AI4S Policy, and Others. Here is the translated text for the remaining categorized papers matching your exact input.)*

### **21. [Chemistry LLM ChemLLM covers 7 million QA data, professional capabilities rival GPT-4](https://hyper.ai/news/34170)**

- **Research highlight:** [https://hyper.ai/news/34170](https://hyper.ai/news/34170)
- **Research Team:** Shanghai AI Lab
- **Related Research:** Large-scale chemical dataset ChemData, ChemPref-10K English/Chinese datasets, C-MHChem dataset, ChemBench4K, ChemBench, Multi-Corpus, NLP tasks.
- **Published Journal:** arXiv, 2024.02
- **Paper Link:** [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852)

### **22. [Wafer-scale producible AI-adaptive micro-spectrometers](https://hyper.ai/news/34075)**

- **Research highlight:** [https://hyper.ai/news/34075](https://hyper.ai/news/34075)
- **Research Team:** Yongfeng Mei's Group at Fudan University
- **Related Research:** Optical spectrometers, miniaturized reconstructive spectrometers, CMOS IC processes, narrow-band channel current datasets.
- **Published Journal:** PNAS, 2024.08
- **Paper Link:** [CMOS-Compatible Reconstructive Spectrometers with Self-Referencing Integrated Fabry-Perot Resonatorsl](https://www.pnas.org/doi/10.1073/pnas.2403950121)

### **23. [GNNOpt model identifies hundreds of solar cell and quantum material candidates](https://hyper.ai/news/35009)**

- **Research highlight:** [https://hyper.ai/news/35009](https://hyper.ai/news/35009)
- **Research Team:** Tohoku University, MIT
- **Related Research:** DFT calculations, GNNOpt, ensemble embeddings, equivariant GNNs, Materials Project database.
- **Published Journal:** Advanced Materials, 2024.06
- **Paper Link:** [Universal Ensemble-Embedding Graph Neural Network for Direct Prediction of Optical Spectra from Crystal Structures](https://onlinelibrary.wiley.com/doi/epdf/10.1002/adma.202409175)

### **24. [Open OMat24 dataset contains 110 million DFT calculation results](https://hyper.ai/news/35515)**

- **Research highlight:** [https://hyper.ai/news/35515](https://hyper.ai/news/35515)
- **Research Team:** Meta
- **Related Research:** Open Materials 2024 (OMat24), EquformerV2 (eqV2), ab initio MD.
- **Published Journal:** arxiv, 2024.10
- **Paper Link:** [Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models](https://arxiv.org/pdf/2410.12771)

### **25. [Novel refractory high-entropy alloy synthesized via machine learning boasts excellent room-temperature ductility](https://hyper.ai/news/35536)**

- **Research highlight:** [https://hyper.ai/news/35536](https://hyper.ai/news/35536)
- **Research Team:** Yanjing Su's Team at University of Science and Technology Beijing
- **Related Research:** ML combined with genetic search, clustering analysis, Multi-Objective Optimization (MOO) frameworks.
- **Published Journal:** Engineering, 2024.09
- **Paper Link:** [Machine-Learning-Assisted Compositional Design of Refractory High-Entropy Alloys with Optimal Strength and Ductility](https://www.sciencedirect.com/science/article/pii/S2095809924005113 )

### **26. [Material generative model FlowLLM features a dataset covering over 45k materials](https://hyper.ai/news/35846)**

- **Research highlight:** [https://hyper.ai/news/35846](https://hyper.ai/news/35846)
- **Research Team:** Meta FAIR, University of Amsterdam
- **Related Research:** FlowLLM, S.U.N. material generation, LLMs, Riemannian Flow Matching (RFM), MP-20 dataset, LoRA.
- **Published Journal:** NeurIPS 2024, 2024.10
- **Paper Link:** [FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions](https://arxiv.org/pdf/2410.23405)

### **27. [Using active learning to identify 14,000 high-entropy oxides, successfully screening 4 high-activity hydrogen evolution catalysts](https://hyper.ai/news/36352)**

- **Research highlight:** [https://hyper.ai/news/36352](https://hyper.ai/news/36352)
- **Research Team:** Xun Wang's Team at Tsinghua, Liang Wu at SJTU, Shengqi Chu at IHEP CAS, Guang Lin at Purdue, Yan Xiang at Duke
- **Related Research:** Active Learning (AL), Kennard-Stone sampling, XRD, CrMnCoNiCu catalysts.
- **Published Journal:** Journal of the American Chemical Society, 2024.10
- **Paper Link:** [Active Learning Guided Discovery of High Entropy Oxides Featuring High H2‑production](https://pubs.acs.org/doi/10.1021/jacs.4c06272)

### **28. [Deep learning model BETE-NET boosts superconducting material search efficiency by 5x](https://hyper.ai/news/37658)**

- **Research highlight:** [https://hyper.ai/news/37658](https://hyper.ai/news/37658)
- **Research Team:** University of Florida, University of Tennessee
- **Related Research:** BETE-NET, α²F(ω) datasets, Eliashberg spectral function datasets.
- **Published Journal:** npj Computational Materials, 2025.01
- **Paper Link:** [Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function](https://www.nature.com/articles/s41524-024-01475-4)

### **29. [Gradient Boosting Decision Tree (GBDT) technology further improves high-precision prediction of high-entropy alloy oxidation resistance](https://hyper.ai/news/37723)**

- **Research highlight:** [https://hyper.ai/news/37723](https://hyper.ai/news/37723)
- **Research Team:** Joint Team from University of Bordeaux, NIMS (Japan), NTHU (Taiwan), KU Leuven, WEL Research Institute
- **Related Research:** GBDT technology, XGBoost algorithm, high-temperature materials, high-entropy alloys (RHEAs and RCCAs).
- **Published Journal:** Scripta Materialia, 2025.01
- **Paper Link:** [Advancing refractory high entropy alloy development with AI-predictive models for high temperature oxidation resistance](https://doi.org/10.1016/j.scriptamat.2024.116394)

### **30. [Molecular design framework RingFormer more precisely predicts organic material molecular optoelectronic properties](https://hyper.ai/news/37870)**

- **Research highlight:** [https://hyper.ai/news/37870](https://hyper.ai/news/37870)
- **Research Team:** The Hong Kong Polytechnic University
- **Related Research:** Molecular design, Transformer architectures, organic solar cells, Graph Neural Networks, RingFormer.
- **Published Journal:** AAAI 2025, 2024.12
- **Paper Link:** [RingFormer: A Ring-Enhanced Graph Transformer for Organic Solar Cell Property Prediction](https://doi.org/10.48550/arXiv.2412.09030)

### **31. [Inorganic retrosynthesis planning method Retrieval-Retro improves inorganic material synthesis efficiency and accuracy](https://hyper.ai/news/37969)**

- **Research highlight:** [https://hyper.ai/news/37969](https://hyper.ai/news/37969)
- **Research Team:** KRICT, KAIST
- **Related Research:** Retrieval-Retro, Convolutional VAEs, masked precursor completion retrievers, neural reaction energy retrievers.
- **Published Journal:** NeurIPS 2024, 2024.10
- **Paper Link:** [Retrieval-Retro: Retrieval-based Inorganic Retrosynthesis with Expert Knowledge](https://doi.org/10.48550/arXiv.2410.21341)

### **32. [Using large models to decipher hydride solid-state electrolyte conduction mechanisms, establishing a reliable activation energy prediction model](https://hyper.ai/news/39173)**

- **Research highlight:** [https://hyper.ai/news/39173](https://hyper.ai/news/39173)
- **Research Team:** Tohoku University, Sichuan University, Shibaura Institute of Technology
- **Related Research:** Solid-state electrolytes (SSEs), LLMs, ab initio metadynamics (MetaD).
- **Published Journal:** Angewandte Chemie-International Edition, 2025.04
- **Paper Link:** [Unraveling the Complexity of Divalent Hydride Electrolytes in Solid-State Batteries via a Data-Driven Framework with Large Language Model](https://go.hyper.ai/isQRi)

### **33. [Tera-scale mass spectrometry data search enabled by machine learning uncovers unknown chemical reactions](https://hyper.ai/news/39224)**

- **Research highlight:** [https://hyper.ai/news/39224](https://hyper.ai/news/39224)
- **Research Team:** Russian Academy of Sciences and others
- **Related Research:** Mass spectrometry, ML-driven search engine MEDUSA Search, PubChem database.
- **Published Journal:** Nature Communications, 2025.01
- **Paper Link:** [Discovering organic reactions with a machine-learning-powered deciphering of tera-scale mass spectrometry data](https://go.hyper.ai/ak7bN)

### **34. [Generative AI structure solution method PXRDnet based on diffusion models successfully solves 200 complex simulated nanocrystals](https://hyper.ai/news/39287)**

- **Research highlight:** [https://hyper.ai/news/39287](https://hyper.ai/news/39287)
- **Research Team:** Columbia University, Stanford University
- **Related Research:** X-ray diffraction, PXRDnet, MP-20-PXRD benchmark dataset, Materials Project database, CDVAE architecture, PXRD regressors.
- **Published Journal:** Nature Materials, 2025.04
- **Paper Link:** [Ab initio structure solutions from nanocrystalline powder diffraction data via diffusion models](https://go.hyper.ai/r1K6b)

### **35. [DreaMS model covers 200 million molecular mass spectra, building the world's largest mass spec dataset GeMS](https://hyper.ai/news/40201)**

- **Research highlight:** [https://hyper.ai/news/40201](https://hyper.ai/news/40201)
- **Research Team:** Institute of Organic Chemistry and Biochemistry, Czech Academy of Sciences
- **Related Research:** GeMS dataset, Locality-Sensitive Hashing (LSH), BERT architectures, self-supervised learning, Fourier features, linear probing.
- **Published Journal:** Nature Biotechnology, 2025.05
- **Paper Link:** [Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS](https://go.hyper.ai/uNbqL)

### **36. [Equivariant machine learning framework accelerates large-scale electric field simulations of materials](https://hyper.ai/news/40600)**

- **Research highlight:** [https://hyper.ai/news/40600](https://hyper.ai/news/40600)
- **Research Team:** Harvard University, Robert Bosch LLC
- **Related Research:** Machine learning frameworks, neural network architectures, material vibrations, dielectric properties, ferroelectric hysteresis.
- **Published Journal:** Nature Communications, 2025.04
- **Paper Link:** [Unified differentiable learning of electric response](https://go.hyper.ai/18TWg)

### **37. [Multi-source data integration method screens 25 types of cement clinker alternatives, equivalent to reducing 1.2 billion tons of greenhouse gases](https://hyper.ai/news/40742)**

- **Research highlight:** [https://hyper.ai/news/40742](https://hyper.ai/news/40742)
- **Research Team:** Soroush Mahjoubi & Elsa A. Olivetti (MIT)
- **Related Research:** LLMs, multi-task neural networks, reactivity evaluation frameworks.
- **Published Journal:** Communication Materials, 2025.05
- **Paper Link:** [Data-driven material screening of secondary and natural cementitious precursors](https://go.hyper.ai/ZOAaW)

### **38. [UNIMATE achieves unified modeling of topology generation/property prediction for the first time](https://hyper.ai/news/41186)**

- **Research highlight:** [https://hyper.ai/news/41186](https://hyper.ai/news/41186)
- **Research Team:** Virginia Tech, Meta AI
- **Related Research:** Metamaterials, 3D topologies, machine learning, UNIMATE model, mechanical metamaterial benchmarks.
- **Published Journal:** ICML 2025, 2025.06
- **Paper Link:** [UNIMATE: A Unified Model for Mechanical Metamaterial Generation, Property Prediction, and Condition Confirmation](https://go.hyper.ai/FoAWw)

### **39. [All-atom diffusion Transformer framework enables unified generation of periodic and aperiodic atomic systems for the first time](https://hyper.ai/news/41503)**

- **Research highlight:** [https://hyper.ai/news/41503](https://hyper.ai/news/41503)
- **Research Team:** Meta FAIR, Cambridge University, MIT
- **Related Research:** Transformers, MP20 dataset, QM9 dataset, GEOM-DRUGS dataset, QMOF dataset.
- **Published Journal:** ICML 2025, 2025.06
- **Paper Link:** [All-atom Diffusion Transformers: Unified generative modelling of molecules and materials](https://go.hyper.ai/27d7U)

### **40. [FASTSOLV model realizes small molecule solubility prediction at any temperature, accelerating inference speed by 50x](https://hyper.ai/news/43318)**

- **Research highlight:** [https://hyper.ai/news/43318](https://hyper.ai/news/43318)
- **Research Team:** MIT Research Team
- **Related Research:** Small molecule solubility prediction, BigSolDB dataset, SolProp dataset, Leeds dataset, FASTSOLV model.
- **Published Journal:** Nature Communication, 2025.08
- **Paper Link:** [Data-driven organic solubility prediction at the limit of aleatoric uncertainty](https://www.nature.com/articles/s41467-025-62717-7)

### **41. [Novel method based on multimodal machine learning models predicts material properties without complete crystal structures](https://hyper.ai/news/43410)**

- **Research highlight:** [https://hyper.ai/news/43410](https://hyper.ai/news/43410)
- **Research Team:** Department of Chemical Engineering & Applied Chemistry, University of Toronto
- **Related Research:** Multimodal machine learning models, CoRE-2019 dataset, BW20K dataset, QMOF dataset, hMOF dataset.
- **Published Journal:** Nature Communications, 2025.07
- **Paper Link:** [Connecting metal-organic framework synthesis to applications using multimodal machine learning](https://www.nature.com/articles/s41467-025-62717-7)

### **42. [AI model CGformer innovatively integrates global attention mechanisms, aiding high-entropy material R&D](https://hyper.ai/news/44908)**

- **Research highlight:** [https://hyper.ai/news/44908](https://hyper.ai/news/44908)
- **Research Team:** Jinjin Li and Fuqiang Huang's Team at AIMS-Lab, SJTU
- **Related Research:** High-entropy materials R&D, AI material design model CGformer, sodium-ion diffusion barrier datasets.
- **Published Journal:** Matter, 2025.08
- **Paper Link:** [CGformer: Transformer-enhanced crystal graph network with global attention for material property prediction](https://www.cell.com/matter/abstract/S2590-2385(25)00423-0)

### **43. [Novel structural constraint integration method SCIGEN adapts to any pre-trained diffusion model](https://hyper.ai/news/44973)**

- **Research highlight:** [https://hyper.ai/news/44973](https://hyper.ai/news/44973)
- **Research Team:** Mingda Li's Team at MIT, Michigan State University, Oak Ridge National Laboratory
- **Related Research:** AL (Archimedean lattices) materials database, diffusion models, crystal structure generation, DiffCSP model.
- **Published Journal:** Nature Materials, 2025.09
- **Paper Link:** [Structural constraint integration in a generative model for the discovery of quantum materials](https://www.nature.com/articles/s41563-025-02355-y)

### **44. [Physically-informed generative AI model SpectroGen requires only single modality input to achieve cross-modal generation with 99% experimental correlation](https://hyper.ai/news/45456)**

- **Research highlight:** [https://hyper.ai/news/45456](https://hyper.ai/news/45456)
- **Research Team:** MIT Research Team
- **Related Research:** SpectroGen, RRUFF database, VAE framework, physical prior models.
- **Published Journal:** Matter, 2025.10
- **Paper Link:** [SpectroGen: A physically informed generative artificial intelligence for accelerated cross-modality spectroscopic materials characterization](https://www.cell.com/matter/abstract/S2590-2385(25)00477-1)

### **45. [MOF-ChemUnity reconstructs MOF panoramic knowledge, pushing material discovery into the "Explainable AI" era](https://hyper.ai/news/46723)**

- **Research highlight:** [https://hyper.ai/news/46723](https://hyper.ai/news/46723)
- **Research Team:** University of Toronto, Clean Energy Innovation Research Centre (NRC Canada)
- **Related Research:** Materials science, MOF-ChemUnity, CoRE MOF 2019 database, QMOF database, LLMs, Graph-augmented RAG.
- **Published Journal:** ACS Publications, 2025.11
- **Paper Link:** [MOF-ChemUnity: Literature-Informed Large Language Models for Metal–Organic Framework Research](https://pubs.acs.org/doi/10.1021/jacs.5c11789)

### **46. [Lightweight universal potential model PET-MAD released, achieving dedicated model-level precision with minimal samples](https://hyper.ai/news/47637)**

- **Research highlight:** [https://hyper.ai/news/47637](https://hyper.ai/news/47637)
- **Research Team:** EPFL
- **Related Research:** First-principles calculations, machine learning interatomic potentials, PET-MAD model, Point Edge Transformer structure.
- **Published Journal:** Nature Communications
- **Paper Link:** [PET-MAD as a lightweight universal interatomic potential for advanced materials modeling](https://www.nature.com/articles/s41467-025-65662-7)

### **47. [AI system ChemOntology released, halving reaction path search costs by integrating chemical knowledge](https://hyper.ai/news/48069)**

- **Research highlight:** [https://hyper.ai/news/48069](https://hyper.ai/news/48069)
- **Research Team:** Hokkaido University
- **Related Research:** Potential Energy Surface (PES), Intrinsic Reaction Coordinates (IRC), Artificial Force Induced Reaction (AFIR), ChemOntology.
- **Published Journal:** ACS Catalysis
- **Paper Link:** [ChemOntology: A Reusable Explicit Chemical Ontology-Based Method to Expedite Reaction Path Searches](https://pubs.acs.org/doi/10.1021/acscatal.5c06298)

### **48. [Princeton and others jointly propose LLM method for predicting MOF free energy, highly accurately assessing synthesis feasibility](https://hyper.ai/news/48685)**

- **Research highlight:** [https://hyper.ai/news/48685](https://hyper.ai/news/48685)
- **Research Team:** Princeton University and Colorado School of Mines
- **Related Research:** Metal-Organic Frameworks (MOFs), free energy prediction, Large Language Models (LLM), thermodynamic evaluation.
- **Published Journal:** JACS (ACS Publications)
- **Paper Link:** [Highly Accurate and Fast Prediction of MOF Free Energy via Machine Learning](https://pubs.acs.org/doi/10.1021/jacs.5c13960)

### **49. [Yale University team proposes MOSAIC model, coordinating LLMs to generate highly reliable chemical synthesis schemes](https://hyper.ai/news/48806)**

- **Research highlight:** [https://hyper.ai/news/48806](https://hyper.ai/news/48806)
- **Research Team:** Yale University Research Team
- **Related Research:** Modern synthetic chemistry, LLMs, MOSAIC model, knowledge structuration.
- **Published Journal:** Nature
- **Paper Link:** [Collective intelligence for AI-assisted chemical synthesiss](https://www.nature.com/articles/s41586-026-10131-4)

### **50. [MIT and others propose diffusion model DiffSyn, enabling generative planning of material synthesis pathways](https://hyper.ai/news/49252)**

- **Research highlight:** [https://hyper.ai/news/49252](https://hyper.ai/news/49252)
- **Research Team:** MIT, Technical University of Munich, and Universitat Politècnica de València
- **Related Research:** Material synthesis planning, generative diffusion model DiffSyn, zeolites.
- **Published Journal:** Nature Computational Science
- **Paper Link:** [DiffSyn: a generative diffusion approach to materials synthesis planning](https://www.nature.com/articles/s43588-025-00949-9)

### **51. [University of Michigan and Farasis Energy jointly propose "Discovery Learning" method, drastically shortening battery life prediction cycles](https://hyper.ai/news/49527)**

- **Research highlight:** [https://hyper.ai/news/49527](https://hyper.ai/news/49527)
- **Research Team:** Prof. Ziyou Song at University of Michigan, Ann Arbor, and Weiran Jiang's Team at Farasis Energy
- **Related Research:** Battery cycle life prediction, Discovery Learning (DL), Scientific Machine Learning, Lithium-ion pouch cell dataset.
- **Published Journal:** Nature
- **Paper Link:** [Discovery Learning predicts battery cycle life from minimal experiments](https://www.nature.com/articles/s41586-025-09951-7)

### **52. [Cornell University proposes SCAN framework, highly accurately predicting and explaining battery electrolyte performance](https://hyper.ai/news/49537)**

- **Research highlight:** [https://hyper.ai/news/49537](https://hyper.ai/news/49537)
- **Research Team:** Cornell University Research Team
- **Related Research:** Salt-solvent chemistry, Non-Aqueous Electrolytes (NAE), SCAN framework, Multi-Feature Network (MFNet), dynamic routing strategy.
- **Published Journal:** Nature Computational Science
- **Paper Link:** [A dynamic routing-guided interpretable framework for salt–solvent chemistry](https://www.nature.com/articles/s43588-026-00955-5)

### **53. [MIT proposes foundation large model DefectNet for non-destructive characterization and quantification of internal material defects](https://hyper.ai/news/50122)**

- **Research highlight:** [https://hyper.ai/news/50122](https://hyper.ai/news/50122)
- **Research Team:** MIT Research Team
- **Related Research:** Materials science, defect engineering, non-destructive characterization, vibrational spectra and Phonon Density of States (PDoS), DefectNet, Machine Learning Interatomic Potentials (MLIPs).
- **Published Journal:** arXiv
- **Paper Link:** [A foundation model for non-destructive defect identification from vibrational spectra](https://arxiv.org/abs/2506.00725)

### **54. [Cornell University proposes multi-agent platform EMSeek, achieving full-pipeline automated analysis of electron microscopy images](https://hyper.ai/news/50298)**

- **Research highlight:** [https://hyper.ai/news/50298](https://hyper.ai/news/50298)
- **Research Team:** Cornell University Research Team
- **Related Research:** Electron Microscopy (EM), multi-agent platform, EMSeek, materials analysis, structural modeling and property inference.
- **Published Journal:** Science Advances
- **Paper Link:** [Bridging electron microscopy and materials analysis with an autonomous agentic platform](https://www.science.org/doi/10.1126/sciadv.aed0583)

## **AI+ Zoology-Botany**

### **1. [SBeA analyzes animal social behaviors based on a few-shot learning framework](https://hyper.ai/news/29353)**

- **Research highlight:** [https://hyper.ai/news/29353](https://hyper.ai/news/29353)
- **Research Team:** Pengfei Wei's Research Team at Shenzhen Institutes of Advanced Technology, CAS
- **Related Research:** PAIR-R24M dataset, bidirectional transfer learning, unsupervised learning, artificial neural networks, identity recognition models. Accuracy in multi-animal identity recognition exceeds 90%.
- **Published Journal:** Nature Machine Intelligence, 2024.01
- **Paper Link:** [Multi-animal 3D social pose estimation, identification and behaviour embedding with a few-shot learning framework](https://www.nature.com/articles/s42256-023-00776-5)

### **2. [Deep learning method based on Siamese networks automatically captures embryonic development processes](https://hyper.ai/news/28419)**

- **Research highlight:** [https://hyper.ai/news/28787](https://hyper.ai/news/28419)
- **Research Team:** Systems Biologist Patrick Müller and University of Konstanz Research Team
- **Related Research:** ImageNet dataset, Siamese networks, deep learning, transfer learning, triplet loss training, iterative training, sub-task training. Identifies key stages of embryonic development without human intervention.
- **Published Journal:** Nature Methods, 2023.11
- **Paper Link:** [Uncovering developmental time and tempo using deep learning](https://www.nature.com/articles/s41592-023-02083-8)

### **3. [Systematic pipeline for collecting plant phenotype data via drones to predict optimal harvest dates](https://hyper.ai/news/28303)**

- **Research highlight:** [https://hyper.ai/news/28303](https://hyper.ai/news/28303)
- **Research Team:** Research Teams from the University of Tokyo and Chiba University
- **Related Research:** Profit prediction models, segmentation models, interactive annotation, LabelMe, non-linear regression models, BiSeNet model.
- **Published Journal:** Plant Phenomics, 2023.09
- **Paper Link:** [Drone-Based Harvest Data Prediction Can Reduce On-Farm Food Loss and Improve Farmer Income](https://spj.science.org/doi/10.34133/plantphenomics.0086#body-ref-B4)

### **4. [AI camera alert system accurately distinguishes tigers from other species](https://hyper.ai/news/27954)**

- **Research highlight:** [https://hyper.ai/news/27954](https://hyper.ai/news/27954)
- **Research Team:** Clemson University Research Team
- **Related Research:** TrailGuard AI. Transmits relevant images to reserve managers' devices within 1 minute.
- **Published Journal:** BioScience, 2023.09
- **Paper Link:** [Accurate proteome-wide missense variant effect prediction with AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492) (Note: Original link provided seems to mismatch the title, but kept as is based on the source text).

### **5. [Using Labrador retriever data and comparing 3 models reveals behavioral traits affecting detection dogs' performance](https://hyper.ai/news/25472)**

- **Research highlight:** [https://hyper.ai/news/25472](https://hyper.ai/news/25472)
- **Research Team:** Abigail Wexner Research Institute at Nationwide Children's Hospital and Rocky Vista University
- **Related Research:** AT tests, Env tests, Random Forest, Support Vector Machines, Logistic Regression, PCA, RFECV.
- **Published Journal:** Scientific Reports, 2023.08
- **Paper Link:** [Machine learning prediction and classification of behavioral selection in a canine olfactory detection program](https://www.nature.com/articles/s41598-023-39112-7)

### **6. [Multi-species image recognition model based on ArcFace Classification Head for face recognition](https://hyper.ai/news/25164)**

- **Research highlight:** [https://hyper.ai/news/25164](https://hyper.ai/news/25164)
- **Research Team:** University of Hawaii Research Team
- **Related Research:** [Cetacean dataset](https://github.com/knshnb/kaggle-happywhale-1st-place), image cropping models, image recognition models, YOLOv5, Detic. Achieved an average accuracy of 0.869.
- **Published Journal:** Methods in Ecology and Evolution, 2023.07
- **Paper Link:** [A deep learning approach to photo–identification demonstrates high performance on two dozen cetacean species](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14167)

### **7. [Monitoring cherry blossom blooming in Japan using Python API and computer vision API](https://hyper.ai/news/24512)**

- **Research highlight:** [https://hyper.ai/news/24512](https://hyper.ai/news/24512)
- **Research Team:** Monash University Research Team (Australia)
- **Related Research:** Social Network Site (SNS) data, Google Cloud Vision AI, machine learning models.
- **Published Journal:** Flora, 2023.07
- **Paper Link:** [The spatiotemporal signature of cherry blossom flowering across Japan revealed via analysis of social network site images](https://www.sciencedirect.com/science/article/abs/pii/S0367253023001019)

### **8. [Machine learning-based population genetics method reveals the formation mechanism of grape flavors](https://hyper.ai/news/24442)**

- **Research highlight:** [https://hyper.ai/news/24442](https://hyper.ai/news/24442)
- **Research Team:** Agricultural Genomics Institute at Shenzhen, CAS
- **Related Research:** [Grapevine genome sequences](https://github.com/zhouyflab/Grapevine_Adaptive_Maladaptive_Introgression), machine learning.
- **Published Journal:** Proceedings of the National Academy of Sciences, 2023.06
- **Paper Link:** [Adaptive and maladaptive introgression in grapevine domestication](https://www.pnas.org/doi/abs/10.1073/pnas.2222041120)

### **9. [Review: Unlocking bioinformatics research more efficiently with AI](https://hyper.ai/news/33931)**

- **Research highlight:** [https://hyper.ai/news/33931](https://hyper.ai/news/33931)
- **Main Content:** AI has abundant application cases in biological fields such as homology search, multiple sequence alignment, phylogenetic construction, genome sequence analysis, and gene discovery. For biological researchers, proficiently integrating machine learning tools into data analysis will undoubtedly accelerate scientific discoveries and improve research efficiency.

### **10. [BirdFlow model accurately predicts flight paths of migratory birds](https://hyper.ai/news/34781)**

- **Research highlight:** [https://hyper.ai/news/33942](https://hyper.ai/news/33942)
- **Research Team:** UMass Amherst, Cornell University
- **Related Research:** Computer modeling, eBird dataset, Markov models, Hyperparameter grid search, Entropy calibration, k-week forecasting.
- **Published Journal:** Methods in Ecology and Evolution, 2023.01
- **Paper Link:** [BirdFlow: Learning seasonal bird movements from eBird data](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14052)

### **11. [New whale bioacoustics model identifies 8 cetacean species](https://hyper.ai/news/34781)**

- **Research highlight:** [https://hyper.ai/news/34781](https://hyper.ai/news/34781)
- **Research Team:** Google Research Team
- **Related Research:** Mel-scale frequency axes, compressed count amplitude, independent invocation via TensorFlow's SavedModel API, Convolutional Neural Networks, classification models for detecting humpback whale calls, interactive visualization tool "Pattern Radio". The model is specifically designed for blue and fin whales and can identify 8 distinct species out of 94 known whale species.
- **Published Journal:** Google Research, 2024.09
- **Paper Link:** [Whistles, songs, boings, and biotwangs: Recognizing whale vocalizations with AI](https://research.google/blog/whistles-songs-boings-and-biotwangs-recognizing-whale-vocalizations-with-ai)

### **12. [Machine learning isolates the sperm whale phonetic alphabet, highly similar to human language with stronger information-carrying capacity](https://hyper.ai/news/33433)**

- **Research highlight:** [https://hyper.ai/news/33433](https://hyper.ai/news/33433)
- **Research Team:** Pratyusha Sharma (MIT) and Project CETI Team
- **Related Research:** DSWP dataset, machine learning, revealing the structural nature of sperm whale vocalizations.
- **Published Journal:** Nature Communications, 2024.05
- **Paper Link:** [Contextual and combinatorial structure in sperm whale vocalisations](https://www.nature.com/articles/s41467-024-47221-8)

### **13. [PlantLncBoost model achieves up to 96% accuracy in cross-species lncRNA prediction](https://hyper.ai/news/40667)**

- **Research highlight:** [https://hyper.ai/news/40667](https://hyper.ai/news/40667)
- **Research Team:** Shandong University of Technology, Beijing Forestry University, Guangdong Academy of Agricultural Sciences, University of São Paulo, Rosalind Franklin University of Medicine and Science, Umeå University
- **Related Research:** GreeNC database, PlantLncBoost algorithm, Random Forest Importance (RFI) strategy, Recursive Feature Elimination (RFE) algorithm.
- **Published Journal:** New Phytologist, 2024.05
- **Paper Link:** [PlantLncBoost: key features for plant lncRNA identification and significant improvement in accuracy and generalization](https://go.hyper.ai/F7pkc)

### **14. [Perch 2.0 covers nearly 15,000 species, refreshing SOTA in bioacoustic classification detection](https://hyper.ai/news/42807)**

- **Research highlight:** [https://hyper.ai/news/42807](https://hyper.ai/news/42807)
- **Research Team:** Google DeepMind, Google Research
- **Related Research:** Bioacoustics, Perch 2.0, Xeno-Canto dataset, iNaturalist dataset, Tierstimmenarchiv dataset, FSD50K dataset, EfficientNet-B3 architecture.
- **Published Journal:** arXiv, 2025.08
- **Paper Link:** [Perch 2.0: The Bittern Lesson for Bioacoustics](https://arxiv.org/abs/2508.04665)

## **AI+ Agriculture-Forestry-Animal husbandry**

### **1. [Using Convolutional Neural Networks for rapid and accurate rice yield estimation](https://hyper.ai/news/26100)**

- **Research highlight:** [https://hyper.ai/news/26100](https://hyper.ai/news/26100)
- **Research Team:** Kyoto University Research Team
- **Related Research:** Convolutional Neural Networks. The CNN model can accurately analyze field photos obtained from different shooting angles, times, and periods, achieving stable yield prediction results.
- **Published Journal:** Plant Phenomics, 2023.07
- **Paper Link:** [Deep Learning Enables Instant and Versatile Estimation of Rice Yield Using Ground-Based RGB Images](https://spj.science.org/doi/10.34133/plantphenomics.0073)

### **2. [Model designed via YOLOv5 algorithm monitors sow posture and piglet birth](https://hyper.ai/news/25131)**

- **Research highlight:** [https://hyper.ai/news/25131](https://hyper.ai/news/25131)
- **Research Team:** Nanjing Agricultural University Research Team
- **Related Research:** YOLOv5, models detecting sow posture and piglets. Can issue alerts 5 hours before farrowing begins with an overall average accuracy of 92.9%.
- **Published Journal:** Sensors, 2023.01
- **Paper Link:** [Sow Farrowing Early Warning and Supervision for Embedded Board Implementations](https://www.mdpi.com/1424-8220/23/2/727)

### **3. [Combining laboratory observation and machine learning to prove that ultrasonic sounds emitted by stressed tomato and tobacco plants can travel in air](https://hyper.ai/news/24547)**

- **Research highlight:** [https://hyper.ai/news/24547](https://hyper.ai/news/24547)
- **Research Team:** Tel Aviv University Research Team (Israel)
- **Related Research:** Machine learning models, SVM, Basic, MFCC, Scattering network, neural network models, leave-one-out cross-validation. Recognition accuracy reached 99.7%; tomato screams peaked on days 4-6.
- **Published Journal:** Cell, 2023.03
- **Paper Link:** [Sounds emitted by plants under stress are airborne and informative](https://doi.org/10.1016/j.cell.2023.03.009)

### **4. [Drone + AI image analysis detects forestry pests](https://hyper.ai/news/23807)**

- **Research highlight:** [https://hyper.ai/news/23807](https://hyper.ai/news/23807)
- **Research Team:** University of Lisbon Research Team
- **Related Research:** FRCNN, YOLO models. The YOLO model exhibited higher detection performance than FRCNN. The combination of drones and AI models can effectively enable early detection of pine processionary moth nests.
- **Published Journal:** NeoBiota, 2023.05
- **Paper Link:** [Testing early detection of pine processionary moth Thaumetopoea pityocampa nests using UAV-based methods](https://neobiota.pensoft.net/article/95692/)

### **5. [Computer vision + deep learning developed for a dairy cow lameness detection system](https://hyper.ai/news/33957)**

- **Research highlight:** [https://hyper.ai/news/33957](https://hyper.ai/news/33957)
- **Research Team:** Newcastle University and Fera Science Ltd. Research Team
- **Related Research:** Computer vision, deep learning, Mask-RCNN algorithms, SORT algorithms, CatBoost algorithms. Accuracy reached 94%-100%.
- **Published Journal:** Nature, 2023.03
- **Paper Link:** [Deep learning pose estimation for multi-cattle lameness detection](https://www.nature.com/articles/s41598-023-31297-1)

## **AI+ Meteorology**

### **1. [Review: Data-driven machine learning weather forecasting models](https://hyper.ai/news/28124)**

- **Research highlight:** [https://hyper.ai/news/28124](https://hyper.ai/news/28124)
- **Main Content:** Numerical Weather Prediction (NWP) is the mainstream method for weather forecasting. It solves the state of the Earth system on a grid-by-grid basis through numerical integration, which is a process of deductive reasoning. Since 2022, machine learning models in weather forecasting have achieved a series of breakthroughs, some of which match the high-precision forecasts of the European Centre for Medium-Range Weather Forecasts (ECMWF).

### **2. [Review: Collecting data from hailstorm centers and predicting extreme weather using large models](https://hyper.ai/news/25874)**

- **Research highlight:** [https://hyper.ai/news/25874](https://hyper.ai/news/25874)
- **Main Content:** In 2021, the Alibaba DAMO Academy and the National Meteorological Center jointly developed an AI algorithm for weather forecasting, successfully predicting multiple severe convective weather events. In September of the same year, DeepMind published a paper in *Nature* using deep generative models for real-time precipitation forecasting.
In early 2023, DeepMind officially launched GraphCast, capable of forecasting the global weather for the next 10 days at a 0.25° resolution within a minute. In April, Nanjing University of Information Science and Technology collaborated with Shanghai AI Laboratory to develop the "FengWu" meteorological large model, further reducing errors compared to GraphCast.
Subsequently, Huawei launched the "Pangu-Weather" large model. By introducing a 3D neural network, Pangu's prediction accuracy surpassed the most accurate NWP forecasting systems for the first time. Recently, Tsinghua University and Fudan University consecutively released the "NowCastNet" and "FuXi" models.

### **3. [Creating new algorithms to accurately predict extreme precipitation using global storm-resolving simulations and machine learning](https://hyper.ai/news/24995)**

- **Research highlight:** [https://hyper.ai/news/24995](https://hyper.ai/news/24995)
- **Research Team:** LEAP Lab at Columbia University
- **Related Research:** Machine learning, Baseline-NN, Org-NN, neural networks.
- **Published Journal:** PNAS, 2023.03
- **Paper Link:** [Implicit learning of convective organization explains precipitation stochasticity](https://www.pnas.org/doi/10.1073/pnas.2216158120)

### **4. [Random Forest-based machine learning model CSU-MLP predicts medium-range severe weather](https://hyper.ai/news/33966)**

- **Research highlight:** [https://hyper.ai/news/33966](https://hyper.ai/news/33966)
- **Research Team:** Colorado State University and NOAA
- **Related Research:** GEFS/R dataset, machine learning, interpolation processing, RF. Capable of accurately predicting severe weather in the medium range (4-8 days).
- **Published Journal:** Weather and Forecasting, 2022.08
- **Paper Link:** [A new paradigm for medium-range severe weather forecasts: probabilistic random forest-based predictions](https://arxiv.org/abs/2208.02383)

### **5. [End-to-end data-driven weather forecasting system Aardvark Weather speeds up predictions by dozens of times compared to traditional methods](https://hyper.ai/news/38605)**

- **Research highlight:** [https://hyper.ai/news/38605](https://hyper.ai/news/38605)
- **Research Team:** Cambridge University, The Alan Turing Institute, University of Toronto, Microsoft Research, ECMWF, British Antarctic Survey, Google DeepMind
- **Related Research:** Weather forecasting systems, HadISD datasets, collaborative microwave-infrared observation networks, ATOVS systems, ASCAT scatterometer data, ERA5 reanalysis datasets, lightweight convolutional networks.
- **Published Journal:** Nature, 2025.03
- **Paper Link:** [End-to-end data-driven weather prediction](https://www.nature.com/articles/s41586-025-08897-0)

### **6. [Machine learning weather forecasting system FCN3 supports ultra-fast single-GPU inference](https://hyper.ai/news/42456)**

- **Research highlight:** [https://hyper.ai/news/42456](https://hyper.ai/news/42456)
- **Research Team:** NVIDIA, Lawrence Berkeley National Laboratory (LBNL), UC Berkeley, Caltech
- **Related Research:** Numerical weather prediction, FourCastNet 3, machine learning, ERA5 dataset, spherical neural operator design, hybrid parallel strategies.
- **Published Journal:** arXiv, 2025.07
- **Paper Link:** [FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale](https://arxiv.org/pdf/2507.12144)

### **7. [Indian monsoon forecasting model based on 36 weather stations achieves city-scale fine forecasting](https://hyper.ai/news/44271)**

- **Research highlight:** [https://hyper.ai/news/44271](https://hyper.ai/news/44271)
- **Research Team:** IIT Bombay, University of Maryland
- **Related Research:** Convolutional Neural Networks (CNN), Transfer Learning (CNN-TL), weather forecasting, Event Synchronization methods, rainfall prediction.
- **Published Journal:** SSRN, 2025.08
- **Paper Link:** [Hyperlocal Extreme Rainfall Forecasts in Mumbai: Convolutional Neural Network Transfer Learning-Based Downscaling Approach](https://go.hyper.ai/j05Vt)

### **8. [ACE2 completes a 4-month seasonal forecast in just 2 minutes](https://hyper.ai/news/44473)**

- **Research highlight:** [https://hyper.ai/news/44473](https://hyper.ai/news/44473)
- **Research Team:** Met Office Hadley Centre, University of Exeter, Allen Institute for AI (Ai2)
- **Related Research:** Seasonal forecasting, ERA5 reanalysis dataset, Global Precipitation Climatology Project (GPCP) v2.3 dataset, ACE2 machine learning atmospheric model.
- **Published Journal:** npj Climate and Atmospheric Science, 2025.08
- **Paper Link:** [Skilful global seasonal predictions from a machine learning weather model trained on reanalysis data](https://go.hyper.ai/YyRfT)

### **9. [Incremental weather forecasting model VA-MoE released, achieving SOTA performance with 75% parameter reduction](https://hyper.ai/news/45152)**

- **Research highlight:** [https://hyper.ai/news/45152](https://hyper.ai/news/45152)
- **Research Team:** HKUST, Zhejiang University, and others
- **Related Research:** Incremental weather forecasting, VA-MoE, ERA5 dataset, two-stage training paradigm, Transformer, multi-task joint loss mechanisms, meteorological forecasting.
- **Published Journal:** ICCV25, 2025.07
- **Paper Link:** [VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](https://arxiv.org/abs/2412.02503)

### **10. [Elucidated Rolling Diffusion Model (ERDM) released, solving long-term forecasting challenges and maintaining a lead over EDM baselines in medium-to-long term forecasts](https://hyper.ai/news/45367)**

- **Research highlight:** [https://hyper.ai/news/45367](https://hyper.ai/news/45367)
- **Research Team:** NVIDIA
- **Related Research:** Medium-range weather forecasting, progressive noise scheduling, Elucidated Diffusion Models (EDM), Elucidated Rolling Diffusion Models (ERDM), Navier-Stokes fluid dynamics benchmark dataset, ERA5 reanalysis dataset, noise scheduling mechanisms, probability flow Ordinary Differential Equations (ODE), denoiser networks.
- **Published Journal:** NeurIPS 2025, 2025.06
- **Paper Link:** [Elucidated Rolling Diffusion Models for Probabilistic Weather Forecasting](https://doi.org/10.48550/arXiv.2506.20024)

### **11. [Novel latent diffusion model OmniCast released, resolving error accumulation in autoregressive weather forecasting models](https://hyper.ai/news/45701)**

- **Research highlight:** [https://hyper.ai/news/45701](https://hyper.ai/news/45701)
- **Research Team:** UCLA Team, Argonne National Laboratory
- **Related Research:** Novel latent diffusion model OmniCast, high-precision probabilistic S2S weather forecasting, Variational Autoencoders (VAE), Transformer models, joint spatial-temporal sampling methods, ERA5 foundation dataset, WeatherBench2 (WB2) test set, ChaosBench test set, UNet architecture.
- **Published Journal:** NeurIPS 2025, 2025.10
- **Paper Link:** [OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](https://go.hyper.ai/YANIu)

### **12. [NVIDIA proposes a novel long-range distillation method, breaking AI bottlenecks in long-term weather forecasting](https://hyper.ai/news/48471)**

- **Research highlight:** [https://hyper.ai/news/48471](https://hyper.ai/news/48471)
- **Research Team:** NVIDIA Research, University of Washington
- **Related Research:** AI weather forecasting models, autoregressive architectures, Subseasonal-to-Seasonal (S2S) forecasting, Long-Range Distillation.
- **Published Journal:** arXiv
- **Paper Link:** [Long-Range Distillation: Distilling 10,000 Years of Simulated Climate into Long Timestep AI Weather Models](https://arxiv.org/abs/2512.22814)

### **13. [Joint team proposes Graph Neural Network model SeaCast, achieving ultra-fast regional ocean forecasting](https://hyper.ai/news/49553)**

- **Research highlight:** [https://hyper.ai/news/49553](https://hyper.ai/news/49553)
- **Research Team:** University of Helsinki, Euro-Mediterranean Center on Climate Change (CMCC), University of Salento
- **Related Research:** Regional ocean forecasting, Graph Neural Networks (GNN), SeaCast model, Mediterranean Forecasting System (MedFS), atmospheric forcing fields.
- **Published Journal:** Scientific Reports
- **Paper Link:** [Accurate Mediterranean Sea forecasting via graph-based deep learning](https://www.nature.com/articles/s41598-025-31177-w)

## **AI+ Astronomy**

### **1. [PRIMO algorithm learns light propagation rules around black holes to reconstruct sharper black hole images](https://hyper.ai/news/23698)**

- **Research highlight:** [https://hyper.ai/news/23698](https://hyper.ai/news/23698)
- **Research Team:** Institute for Advanced Study (Princeton)
- **Related Research:** PRIMO algorithm, PCA, GRMHD. PRIMO reconstructed the black hole image.
- **Published Journal:** The Astrophysical Journal Letters, 2023.04
- **Paper Link:** [The Image of the M87 Black Hole Reconstructed with PRIMO](https://iopscience.iop.org/article/10.3847/2041-8213/acc32d/pdf)

### **2. [Training computer vision algorithms with simulated data to sharpen and "restore" astronomical images](https://hyper.ai/news/33975)**

- **Research highlight:** [https://hyper.ai/news/33975](https://hyper.ai/news/33975)
- **Research Team:** Tsinghua University and Northwestern University
- **Related Research:** [GalSim](https://github.com/GalSim-developers/GalSim), [COSMOS](https://doi.org/10.5281/zenodo.3242143), computer vision algorithms, CNNs, Richardson-Lucy algorithm, unrolled-ADMM neural networks.
- **Published Journal:** Monthly Notices of the Royal Astronomical Society, 2023.06
- **Paper Link:** [Galaxy image deconvolution for weak gravitational lensing with unrolled plug-and-play ADMM](https://www.nature.com/articles/s41421-023-00543-1)

### **3. [Using unsupervised machine learning algorithm Astronomaly to find previously overlooked anomalies](https://hyper.ai/news/26316)**

- **Research highlight:** [https://hyper.ai/news/26316](https://hyper.ai/news/26316)
- **Research Team:** Researchers at the University of the Western Cape (UWC)
- **Related Research:** CNN, unsupervised machine learning, Astronomaly, PCA, Isolation Forest, LOF algorithm, iForest algorithm, NS algorithm, DR algorithm. Astronomaly found 1,635 anomalies out of the 2,000 images with the highest anomaly scores.
- **Published Journal:** arXiv, 2023.09
- **Paper Link:** [Astronomaly at Scale: Searching for Anomalies Amongst 4 Million Galaxies](https://arxiv.org/abs/2309.08660)

### **4. [Machine learning-based method for CME identification and parameter extraction](https://hyper.ai/news/31870)**

- **Research highlight:** [https://hyper.ai/news/31870](https://hyper.ai/news/31870)
- **Research Team:** State Key Laboratory of Space Weather, National Space Science Center, CAS
- **Related Research:** Machine learning, neural networks, Otsu algorithm, trajectory matching algorithms, automated identification, parameter extraction, CACTus, CORIMP, SEEDS. Capable of identifying Coronal Mass Ejections.
- **Published Journal:** THE ASTROPHYSICAL JOURNAL, 2024.04
- **Paper Link:** [An Algorithm for the Determination of Coronal Mass Ejection Kinematic Parameters Based on Machine Learning](https://iopscience.iop.org/article/10.3847/1538-4365/ad2dea)

### **5. [Deep learning discovers 107 cases of neutral carbon absorption lines](https://hyper.ai/news/32210)**

- **Research highlight:** [https://hyper.ai/news/32210](https://hyper.ai/news/32210)
- **Research Team:** International team led by Researcher Jian Ge at Shanghai Astronomical Observatory, CAS
- **Related Research:** Deep learning methods, SDSS DR12, Convolutional Neural Network models. Discovered 107 cases of neutral atomic-carbon absorbers in the early universe, with a detection precision of 99.8%.
- **Published Journal:** MNRAS, 2024.05
- **Paper Link:** [Detecting rare neutral atomic-carbon absorbers with a deep neural network](https://doi.org/10.1093/mnras/stae799)

### **6. [StarFusion model achieves high spatial resolution image prediction](https://hyper.ai/news/34254)**

- **Research highlight:** [https://hyper.ai/news/34254](https://hyper.ai/news/34254)
- **Research Team:** Jin Chen's Team at the State Key Laboratory of Earth Surface Processes and Resource Ecology, BNU
- **Related Research:** Deep learning methods, remote sensing imagery, high spatial resolution image prediction, proposed dual-stream spatiotemporal decoupled fusion architecture model StarFusion, Gaofen-1 datasets, Sentinel-2 satellite datasets, SRGAN-STF model, linear regression models, multivariate regression models.
- **Published Journal:** Journal of Remote Sensing, 2024.07
- **Paper Link:** [A Hybrid Spatiotemporal Fusion Method for High Spatial Resolution Imagery: Fusion of Gaofen-1 and Sentinel-2 over Agricultural Landscapes](https://spj.science.org/doi/10.34133/remotesensing.0159)

### **7. [Satellite image generation method developed based on SD3, constructing the largest remote sensing dataset to date, EcoMapper](https://hyper.ai/news/41041)**

- **Research highlight:** [https://hyper.ai/news/41041](https://hyper.ai/news/41041)
- **Research Team:** Technical University of Munich, University of Zurich
- **Related Research:** Remote sensing dataset EcoMapper, Stable Diffusion 3, DiffusionSat, multi-conditional image generation, satellite image generation.
- **Published Journal:** ICML 2025, 2024.06
- **Paper Link:** [EcoMapper: Generative Modeling for Climate-Aware Satellite Imagery](https://go.hyper.ai/VFRWu)

### **8. [Geospatial AI Earth AI focuses on 3 core data types, improving geospatial reasoning capabilities by 64%](https://hyper.ai/news/45528)**

- **Research highlight:** [https://hyper.ai/news/45528](https://hyper.ai/news/45528)
- **Research Team:** Google Research, Google X, Google Cloud
- **Related Research:** Geospatial AI, RS-Landmarks dataset, RS-WebLI dataset, RS-Global dataset, Earth AI, Foundation Models (FMs), Large Language Models (LLM), remote sensing foundation models, spatial alignment + representation integration, geospatial reasoning.
- **Published Journal:** arXiv, 2024.10
- **Paper Link:** [Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning](https://doi.org/10.48550/arXiv.2510.18318)

### **9. [The first astronomical multimodal foundation model AION-1 is born, pre-trained on 200 million astronomical targets](https://hyper.ai/news/46802)**

- **Research highlight:** [https://hyper.ai/news/46802](https://hyper.ai/news/46802)
- **Research Team:** UC Berkeley, Cambridge, Oxford, and teams from over 10 global research institutions
- **Related Research:** AION-1, multimodal cosmological datasets, Tokenization schemes, Transformer encoder-decoder structure, ResNet structure.
- **Published Journal:** NeurIPS 2025, 2025.10
- **Paper Link:** [AION-1: Omnimodal Foundation Model for Astronomical Sciences](ttps://openreview.net/forum?id=6gJ2ZykQ5W)

### **10. [Novel data-driven pipeline precisely identifies 7 rare lensed samples from 810,000 quasars using CNN](https://hyper.ai/news/47240)**

- **Research highlight:** [https://hyper.ai/news/47240](https://hyper.ai/news/47240)
- **Research Team:** Stanford, SLAC National Accelerator Laboratory, Peking University, INAF - Brera Astronomical Observatory, UCL, UC Berkeley, etc.
- **Related Research:** Convolutional Neural Networks (CNN), DESI datasets, strong gravitational lenses, quasars, black hole research, galactic co-evolution, FastSpec catalogs.
- **Published Journal:** arXiv, 2024.10
- **Paper Link:** [Quasars acting as Strong Lenses Found in DESI DR1](https://arxiv.org/abs/2511.02009)

### **11. [ESA team proposes semi-supervised method AnomalyMatch to efficiently screen rare celestial bodies from nearly 100 million Hubble records](https://hyper.ai/news/49138)**

- **Research highlight:** [https://hyper.ai/news/49138](https://hyper.ai/news/49138)
- **Research Team:** European Space Astronomy Centre (ESAC) under the European Space Agency (ESA)
- **Related Research:** Astrophysical anomalies, semi-supervised binary classification, active learning, AnomalyMatch, Hubble Legacy Archive.
- **Published Journal:** Astronomy & Astrophysics
- **Paper Link:** [Identifying astrophysical anomalies in 99.6 million source cutouts from the Hubble legacy archive using AnomalyMatch](https://doi.org/10.1051/0004-6361/202555512)

### **12. [University of Warwick proposes the RAVEN validation pipeline, confirming 118 new exoplanets](https://hyper.ai/news/50073)**

- **Research highlight:** [https://hyper.ai/news/50073](https://hyper.ai/news/50073)
- **Research Team:** University of Warwick Research Team
- **Related Research:** Exoplanet validation, Transiting Exoplanet Survey Satellite (TESS), RAVEN pipeline, synthetic training datasets, false-positive elimination.
- **Published Journal:** arXiv
- **Paper Link:** [RAVEN: RAnking and Validation of ExoplaNets](https://arxiv.org/abs/2509.17645)

### **13. [University of Warwick proposes an ensemble learning framework to highly accurately predict asteroseismic parameters for δ Scuti stars](https://hyper.ai/news/50946)**

- **Research highlight:** [https://hyper.ai/news/50946](https://hyper.ai/news/50946)
- **Research Team:** University of Warwick Research Team
- **Related Research:** δ Scuti stars, asteroseismology, TESS light curve data, ensemble machine learning frameworks, large frequency separation Δν.
- **Published Journal:** The Astronomical Journal
- **Paper Link:** [Ensemble Machine Learning Approach to Estimate the Asteroseismic Indices for δ Scuti Stars Observed by TESS](https://beta.iopscience.iop.org/article/10.3847/1538-3881/ae4bd8)

### **14. [Spanish research team proposes the StreakMind system, utilizing AI to automatically detect satellite streaks in astronomical images](https://hyper.ai/news/51385)**

- **Research highlight:** [https://hyper.ai/news/51385](https://hyper.ai/news/51385)
- **Research Team:** Spanish Royal Naval Observatory (ROA) and other institutions
- **Related Research:** Near-Earth Object (NEO) detection, planetary defense, astronomical image streak detection, StreakMind system, YOLO11.
- **Published Journal:** arXiv
- **Paper Link:** [StreakMind: AI detection and analysis of satellite streaks in astronomical images with automated database integration](https://hyper.ai/papers/2605.03429)

## **AI+ Natural Disaster**

### **1. [Machine learning predicts land subsidence risk over the next 40 years](https://hyper.ai/news/30173)**

- **Research highlight:** [https://hyper.ai/news/30173](https://hyper.ai/news/30173)
- **Research Team:** Jianxin Liu's Research Team at Central South University
- **Related Research:** SAR datasets, machine learning models, XGBR, LSTM.
- **Published Journal:** Journal of Environmental Management, 2024.02
- **Paper Link:** [Machine learning-based techniques for land subsidence simulation in an urban area](https://www.sciencedirect.com/science/article/abs/pii/S0301479724000641?via%3Dihub)

### **2. [Semantic segmentation model SCDUNet++ used for landslide mapping](https://hyper.ai/news/29672)**

- **Research highlight:** [https://hyper.ai/news/29672](https://hyper.ai/news/29672)
- **Research Team:** Rui Liu's Research Team at Chengdu University of Technology
- **Related Research:** Sentinel-2 multispectral data, NASADEM data, landslide data, GLFE, CNN, DSSA, DSC, DTL, Transformer, deep transfer learning. Intersection over Union (IoU) increased by 1.91% - 24.42%, and F1 increased by 1.26% - 18.54%.
- **Published Journal:** International Journal of Applied Earth Observation and Geoinformation, 2024.01
- **Paper Link:** [A deep learning system for predicting time to progression of diabetic retinopathy](https://www.nature.com/articles/s41591-023-02702-z) *(Note: Link mismatch present in source, kept as is).*

### **3. [Neural networks convert 2D solar images into 3D reconstructed images](https://hyper.ai/news/28797)**

- **Research highlight:** [https://hyper.ai/news/28797](https://hyper.ai/news/28797)
- **Research Team:** National Center for Atmospheric Research (NCAR)
- **Related Research:** NeRFs neural networks, SuNeRF model. Revealed the sun's poles for the first time.
- **Published Journal:** arxiv, 2022.11
- **Paper Link:** [SuNeRF: Validation of a 3D Global Reconstruction of the Solar Corona Using Simulated EUV Images](https://arxiv.org/abs/2211.14879)

### **4. [Additive neural networks analyze influencing factors in natural disasters](https://hyper.ai/news/24957)**

- **Research highlight:** [https://hyper.ai/news/24957](https://hyper.ai/news/24957)
- **Research Team:** UCLA Research Team
- **Related Research:** Additive neural networks, semi-automatic detection algorithms, additive ANN, SNN, feature selection models, multi-stage training.
- **Published Journal:** Communications Earth & Environment, 2023.05
- **Paper Link:** [Landslide susceptibility modeling by interpretable neural network](https://www.nature.com/articles/s43247-023-00806-5)

### **5. [Using Explainable AI to analyze various geographical factors in Gippsland, Australia](https://hyper.ai/news/33994)**

- **Research highlight:** [https://hyper.ai/news/33994](https://hyper.ai/news/33994)
- **Research Team:** Australian National University, University of Technology Sydney
- **Related Research:** Random Forest models, machine learning models, cross-validation techniques. XAI can effectively predict wildfire occurrences based on geographical features.
- **Published Journal:** ScienceDirect, 2023.06
- **Paper Link:** [Explainable artificial intelligence (XAI) for interpreting the contributing factors feed into the wildfire susceptibility prediction model](https://www.sciencedirect.com/science/article/pii/S0048969723016224)

### **6. [Machine learning-based flood forecasting model](https://hyper.ai/news/31060)**

- **Research highlight:** [https://hyper.ai/news/31060](https://hyper.ai/news/31060)
- **Research Team:** Google Research
- **Related Research:** HydroATLAS project, LSTM networks, encoder-decoders, cross-validation. Performance exceeded state-of-the-art GloFAS forecasting models.
- **Published Journal:** Nature, 2024.03
- **Paper Link:** [Global prediction of extreme floods in ungauged watersheds](https://www.nature.com/articles/s41586-024-07145-1)

### **7. [ED-DLSTM achieves flood prediction in unmonitored areas](https://hyper.ai/news/32138)**

- **Research highlight:** [https://hyper.ai/news/32138](https://hyper.ai/news/32138)
- **Research Team:** Chaojun Ouyang's Team at the Institute of Mountain Hazards and Environment (IMHE), CAS
- **Related Research:** Data from 2,000 hydrological stations, training datasets from the US, UK, Central Europe, Canada, cross-region spatiotemporal ensemble models, encoder-decoders, multimodal data, spatial static grid attribute data, residual convolutions.
- **Published Journal:** The Innovation, 2024.04
- **Paper Link:** [Deep learning for cross-region streamflow and flood forecasting at a global scale](https://doi.org/10.1016/j.xinn.2024.100617)

### **8. [ChloroFormer model provides early warning of marine algal blooms](https://hyper.ai/news/34544)**

- **Research highlight:** [https://hyper.ai/news/34544](https://hyper.ai/news/34544)
- **Research Team:** GIS Lab at Zhejiang University
- **Related Research:** TZ02 dataset, deep learning model ChloroFormer, Transformer neural networks, frequency filter mechanisms, frequency attention mechanisms. ChloroFormer surpassed baselines in short-term and medium-term Chlorophyll-a predictions.
- **Published Journal:** Water Research, 2024.10
- **Paper Link:** [Enhanced forecasting of chlorophyll-a concentration in coastal waters through integration of Fourier analysis and Transformer networks](https://doi.org/10.1016/j.watres.2024.122160 )

### **9. [The first marine large language model OceanGPT accepted by ACL 2024! Underwater embodied AI becomes reality](https://hyper.ai/news/33044)**

- **Research highlight:** [https://hyper.ai/news/33044](https://hyper.ai/news/33044)
- **Research Team:** Ningyu Zhang and Huajun Chen's Team, College of Computer Science and Technology, Zhejiang University
- **Related Research:** Marine-domain LLMs, regular expressions, Hash algorithms, marine science instruction generation framework DoInstruct, multi-agent collaboration, gpt-3.5-turbo, BM25 algorithms, LLaMA-2, Vicuna-7b-1.5, embodied AI.
- **Published Journal:** ACL 2024, 2024.05
- **Paper Link:** [OceanGPT: A Large Language Model for Ocean Science Tasks](https://arxiv.org/abs/2310.02031)

### **10. [AI predicts global warming trends](https://hyper.ai/news/36778)**

- **Research highlight:** [https://hyper.ai/news/36778](https://hyper.ai/news/36778)
- **Research Team:** Joint Research Team from Stanford University, Colorado State University, and ETH Zurich
- **Related Research:** AI CNN systems, global climate models, transfer learning, predicting conditions under continuously increasing carbon emissions, verifying the accuracy of predictive frameworks across different historical periods. AI predicts a 90% probability of record-breaking maximum temperature shifts.
- **Published Journal:** Geophysical Research Letters, 2024.12
- **Paper Link:** [Data-Driven Predictions of Peak Warming Under Rapid Decarbonization](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL111832)

### **11. [New GeoAI model explains surface heat flow distribution on the Tibetan Plateau](https://hyper.ai/news/36501)**

- **Research highlight:** [https://hyper.ai/news/36501](https://hyper.ai/news/36501)
- **Research Team:** School of Earth Sciences, Zhejiang University
- **Related Research:** Spatial intelligence methods—Explainable-enhanced Geographically Neural Network Weighted Regression (EI-GNNWR) model, surface heat flow datasets, NGHF continental heat flow datasets, Chinese continental surface heat flow datasets, SHAP value calculations, Extreme Gradient Boosting models, fully connected neural network models, Ordinary Least Squares, Geographically Weighted Regression models.
- **Published Journal:** Journal of Geophysical Research: Solid Earth, 2024.10
- **Paper Link:** [The Distribution of Surface Heat Flow on the Tibetan Plateau Revealed by Data‐Driven Methods](https://doi.org/10.1029/2023JB028491)

### **12. ["WenHai" marine environment intelligent forecasting large model outperforms numerical marine forecasting](https://hyper.ai/news/38294)**

- **Research highlight:** [https://hyper.ai/news/38294](https://hyper.ai/news/38294)
- **Research Team:** Research Team led by Academician Lixin Wu at Laoshan Laboratory, OUC, USTC, Qingdao Guoshi Technology Group
- **Related Research:** Marine environmental forecasting, physical oceanography, artificial intelligence, marine dynamics theory-driven neural network architecture design, explicit embedding of bulk formulas into neural networks.
- **Published Journal:** Nature Communications, 2025.03
- **Paper Link:** [Forecasting the Eddying Ocean with a Deep Neural Network](https://www.nature.com/articles/s41467-025-57389-2)

### **13. [University of Minnesota proposes knowledge-guided machine learning model FHNN, realizing high-precision flood forecasting](https://hyper.ai/news/49992)**

- **Research highlight:** [https://hyper.ai/news/49992](https://hyper.ai/news/49992)
- **Research Team:** University of Minnesota Twin Cities Research Team
- **Related Research:** Flood forecasting, Knowledge-Guided Machine Learning (KGML), Factorized Hierarchical Neural Networks (FHNN), Process-Based Models (PBM), hydrological cycles, and runoff prediction.
- **Published Journal:** Water Resources Research
- **Paper Link:** [Knowledge-Guided Machine Learning for Operational Flood Forecasting](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR039064)

### **14. [Google releases version 2 of its global flood forecasting system, significantly extending valid forecast times](https://hyper.ai/news/51472)**

- **Research highlight:** [https://hyper.ai/news/51472](https://hyper.ai/news/51472)
- **Research Team:** Google Research
- **Related Research:** Flood forecasting, hydrological simulation, machine learning hydrological models, Global Flood Forecasting Model v2, Google Runoff Reanalysis and Reforecasts (GRRR) dataset.
- **Published Journal:** EGUsphere
- **Paper Link:** [Extending Medium-Range Global Flood Forecasts: The Google Global Flood Forecasting Model Version 2](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-2283/)

## **Others**

### **1. [TacticAI football assistant hits 90% practical utility in tactical layouts](https://hyper.ai/news/30454)**

- **Research highlight:** [https://hyper.ai/news/30454](https://hyper.ai/news/30454)
- **Research Team:** Google DeepMind and Liverpool FC
- **Related Research:** Geometric deep learning, GNNs, predictive models, generative models. Increased shooting opportunities by 13%.
- **Published Journal:** Nature, 2024.03
- **Paper Link:** [TacticAI: an AI assistant for football tactics](https://www.nature.com/articles/s41467-024-45965-x)

### **2. [Denoising diffusion model SPDiff enables long-range crowd movement simulation](https://hyper.ai/news/30069)**

- **Research highlight:** [https://hyper.ai/news/30069](https://hyper.ai/news/30069)
- **Research Team:** Center for Urban Science and Computation (EE Dept, Tsinghua), Shenzhen Key Laboratory of Ubiquitous Data Enabling (Tsinghua SIGS), Peng Cheng Laboratory
- **Related Research:** GC dataset, UCY dataset, conditional denoising diffusion models, SPDiff, GN, EGCL, LSTM, multi-frame rollout training algorithms. Reached optimal performance with only 5% of training data.
- **Published Journal:** Nature, 2024.02
- **Paper Link:** [Social Physics Informed Diffusion Model for Crowd Simulation](https://arxiv.org/abs/2402.06680)

### **3. [Intelligent scientific facilities drive paradigm shifts in research](https://hyper.ai/news/29570)**

- **Research highlight:** [https://hyper.ai/news/29570](https://hyper.ai/news/29570)
- **Research Team:** Hong Mei's Research Team at Shanghai Jiao Tong University
- **Related Research:** Scientific large models, generative simulation and inversion, autonomous intelligent unmanned experiments, large-scale trusted scientific collaboration, AI research assistants.
- **Published Journal:** Bulletin of Chinese Academy of Sciences, 2023.12
- **Paper Link:** [AI for Science: Intelligent scientific facilities revolutionize fundamental research](http://www.bulletin.cas.cn/previewFile?id=52965146&type=pdf&lang=zh)

### **4. [DeepSymNet represents symbolic expressions based on supervised learning](https://hyper.ai/news/29243)**

- **Research highlight:** [https://hyper.ai/news/29243](https://hyper.ai/news/29243)
- **Research Team:** Min Wu's Research Team at the Institute of Semiconductors, CAS
- **Related Research:** [Symbolic network datasets](https://hyper.ai/datasets/29321), DSNOrg, DSNB, DSNBM, supervised learning. Uses shorter labels, reduces the search space for predictions, and enhances algorithm robustness.
- **Published Journal:** Journals & Magazines, 2023.11
- **Paper Link:** [Discovering Mathematical Expressions Through DeepSymNet: A Classification-Based Symbolic Regression Framework](https://ieeexplore.ieee.org/document/10327762)

### **5. [Large language model ChipNeMo assists engineers in chip design](https://hyper.ai/news/29134)**

- **Research highlight:** [https://hyper.ai/news/29134](https://hyper.ai/news/29134)
- **Research Team:** NVIDIA Research Team
- **Related Research:** Domain adaptation techniques, NVIDIA NeMo, domain-adapted retrieval models, RAG, supervised fine-tuning with domain-specific instructions, DAPT, SFT, Tevatron, LLMs.
- **Published Journal:** arXiv, 2024.04
- **Paper Link:** [ChipNeMo: Domain-Adapted LLMs for Chip Design](https://arxiv.org/abs/2311.00176)

### **6. [AlphaGeometry can solve geometry problems](https://hyper.ai/news/29059)**

- **Research highlight:** [https://hyper.ai/news/29059](https://hyper.ai/news/29059)
- **Research Team:** Google DeepMind Research Team
- **Related Research:** Neural language models, symbolic deduction engines, language models.
- **Published Journal:** Nature, 2024.01
- **Paper Link:** [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)

### **7. [Reinforcement learning applied to urban spatial planning](https://hyper.ai/news/28917)**

- **Research highlight:** [https://hyper.ai/news/28917](https://hyper.ai/news/28917)
- **Research Team:** Yong Li's Research Team at Tsinghua University
- **Related Research:** Deep reinforcement learning, human–artificial intelligence collaborative frameworks, urban planning models, policy networks, value networks, GNNs. Defeated 8 professional human planners on service and ecological metrics.
- **Published Journal:** Nature Computational Science, 2023.09
- **Paper Link:** [Spatial planning of urban communities via deep reinforcement learning](https://www.nature.com/articles/s43588-023-00503-5)

### **8. [ChatArena framework: Playing Werewolf with Large Language Models](https://hyper.ai/news/28576)**

- **Research highlight:** [https://hyper.ai/news/28576](https://hyper.ai/news/28576)
- **Research Team:** Peng Li's Research Team at Tsinghua University
- **Related Research:** Non-parametric learning mechanisms, language models, Prompts.
- **Published Journal:** arxiv, 2023.09
- **Paper Link:** [Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf](https://arxiv.org/pdf/2309.04658.pdf)

### **9. [Review: 30 scholars co-publish in Nature, 10-year retrospective deconstructs how AI reshapes scientific paradigms](https://hyper.ai/news/28166)**

- **Research highlight:** [https://hyper.ai/news/28166](https://hyper.ai/news/28166)
- **Main Content:** Postdoc Hanchen Wang from Stanford's Computer Science and Genetics, along with Tianfan Fu from Georgia Tech CSE, Yuanqi Du from Cornell CS, and 27 others, reviewed the role of AI in fundamental scientific research over the past decade and outlined persisting challenges and shortcomings.
- **Paper Link:** [Scientific discovery in the age of artificial intelligence](https://www.nature.com/articles/s41586-023-06221-2)

### **10. [Ithaca assists epigraphers in text restoration, chronological attribution, and geographical attribution](https://hyper.ai/news/28140)**

- **Research highlight:** [https://hyper.ai/news/28140](https://hyper.ai/news/28140)
- **Research Team:** DeepMind and Ca' Foscari University of Venice
- **Related Research:** I.PHI dataset, Ithaca model, Kullback-Leibler divergence, cross-entropy loss functions. Text restoration accuracy hit 62%, chronological attribution error within 30 years, and geographical attribution accuracy reached 71%.
- **Published Journal:** Nature, 2020.03
- **Paper Link:** [Restoring and attributing ancient texts using deep neural networks](https://www.nature.com/articles/s41586-022-04448-z)

### **11. [AI in forward and inverse problems of meta-optics, data analysis based on metasurface systems](https://hyper.ai/news/34006)**

- **Research highlight:** [https://hyper.ai/news/34006](https://hyper.ai/news/34006)
- **Research Team:** City University of Hong Kong
- **Related Research:** Predicting NNs, Deep Neural Networks. Prediction accuracy exceeded 99%.
- **Published Journal:** ACS Publications, 2022.06
- **Paper Link:** [Artificial Intelligence in Meta-optics](https://pubs.acs.org/doi/10.1021/acs.chemrev.2c00012)

### **12. [A new geospatial artificial intelligence method: Geographically Neural Network Weighted Logistic Regression](https://hyper.ai/news/30608)**

- **Research highlight:** [https://hyper.ai/news/30608](https://hyper.ai/news/30608)
- **Research Team:** Zhenhong Du's Research Team at Zhejiang University
- **Related Research:** Spatial patterns, neural networks, Shapley Additive Explanations (SHAP), Inverse Distance Weighting interpolation, binary cross-entropy loss functions, 5-fold cross-validation. Outperformed other advanced models in mineral prospectivity mapping.
- **Published Journal:** International Journal of Applied Earth Observation and Geoinformation, 2024.04
- **Paper Link:** [Enhancing mineral prospectivity mapping with geospatial artificial intelligence: A geographically neural network-weighted logistic regression approach](https://doi.org/10.1016/j.jag.2024.103746)

### **13. [Using diffusion models to generate neural network parameters, transforming spatiotemporal few-shot learning into a diffusion model pre-training problem](https://hyper.ai/news/30545)**

- **Research highlight:** [https://hyper.ai/news/30545](https://hyper.ai/news/30545)
- **Research Team:** Yong Li's Research Team at the Center for Urban Science and Computation, EE Dept, Tsinghua University
- **Related Research:** Smart cities, spatiotemporal data, knowledge transfer, MetaLA, PEMS-BAy, Transformer diffusion models, conditional generation framework GPD, neural networks, neural network parameters, pre-training + prompt tuning.
- **Published Journal:** ICLR 2024, 2024.01
- **Paper Link:** [Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation](https://openreview.net/forum?id=QyFm3D3Tzi)

### **14. [Latest AI4S insights from Fei-Fei Li's team: 16 innovative technologies summarized, covering biology/materials/healthcare/diagnostics](https://hyper.ai/news/31499)**

- **Research highlight:** [https://hyper.ai/news/31499](https://hyper.ai/news/31499)
- **Main Content:** Stanford's HAI released the "2024 AI Index Report," comprehensively tracking global AI development trends in 2023. It also explored AI's profound impact in science and medicine, highlighting 2023's brilliant AI achievements in science and breakthrough medical innovations like SynthSR and ImmunoSEIRA. Furthermore, it analyzed FDA trends in AI medical device approvals, providing valuable reference for the industry.

### **15. [Accurate prediction of Wuhan housing prices! osp-GNNWR model accurately describes complex spatial processes and geographical phenomena](https://hyper.ai/news/32453)**

- **Research highlight:** [https://hyper.ai/news/32453](https://hyper.ai/news/32453)
- **Research Team:** Sensen Wu's Team at the GIS Lab, Zhejiang University
- **Related Research:** Neural networks, spatial proximity optimization, Geographically Neural Network Weighted Regression methods, dataset of 968 Anjuke real estate samples, spatial regression models, gradient descent algorithms.
- **Published Journal:** International Journal of Geographical Information Science, 2024.04
- **Paper Link:** [A neural network model to optimize the measure of spatial proximity in geographically weighted regression approach: a case study on house price in Wuhan](https://www.tandfonline.com/doi/abs/10.1080/13658816.2024.2343771)

### **16. [Introducing zero-shot learning to release a conditional diffusion model optimized for oracle bone script decipherment](https://hyper.ai/news/33010)**

- **Research highlight:** [https://hyper.ai/news/33010](https://hyper.ai/news/33010)
- **Research Team:** Xiang Bai and Yuliang Liu's Team at HUST, jointly with University of Adelaide, Anyang Normal University, SCUT
- **Related Research:** Conditional diffusion models, image generation techniques, local analytic sampling techniques, HUST-OBS dataset, EVOBC dataset, ResNet-101 backbones, OCR technology, zero-shot learning strategies, style encoders, content encoders.
- **Published Journal:** ACL 2024, 2024.06
- **Paper Link:** [Deciphering Oracle Bone Language with Diffusion Models](https://doi.org/10.48550/arXiv.2406.00684)

### **17. [Stanford/Apple and 23 other institutions release the DCLM benchmark; foundation model performs on par with Llama3 8B](https://hyper.ai/news/33001)**

- **Research highlight:** [https://hyper.ai/news/33001](https://hyper.ai/news/33001)
- **Research Team:** Joint effort by UW, Stanford, Apple, and 20 other institutions
- **Related Research:** Language models, DCLM benchmark, Transformers, MMLU.
- **Published Journal:** arXiv, 2024.06
- **Paper Link:** [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)

### **18. [PoCo solves the data source heterogeneity dilemma, enabling robots to execute multi-tasks flexibly](https://hyper.ai/news/32765)**

- **Research highlight:** [https://hyper.ai/news/32765](https://hyper.ai/news/32765)
- **Research Team:** MIT Researchers
- **Related Research:** Denoising Diffusion Probabilistic Models (DDPM), Denoising Diffusion Implicit Models (DDIM), probabilistic composition of diffusion models, robotic policy composition framework PoCo.
- **Published Journal:** arXiv, 2024.05
- **Paper Link:** [PoCo: Policy Composition from and for Heterogeneous Robot Learning](https://arxiv.org/abs/2402.02511)

### **19. [Containing 140,000 images! Oracle bone script dataset helps team win ACL Best Paper Award](https://hyper.ai/news/33826)**

- **Research highlight:** [https://hyper.ai/news/33826](https://hyper.ai/news/33826)
- **Research Team:** Prof. Xiang Bai's Research Team at HUST
- **Related Research:** HUST-OBC dataset, unsupervised visual contrastive learning models.
- **Published Journal:** Scientific Data, 2024.06
- **Paper Link:** [An open dataset for oracle bone script recognition and decipherment](https://arxiv.org/abs/2401.15365)

### **20. [Proposing a channel prediction scheme based on pre-trained LLMs, GPT-2 empowers the physical layer of wireless communications](https://hyper.ai/news/33195)**

- **Research highlight:** [https://hyper.ai/news/33195](https://hyper.ai/news/33195)
- **Research Team:** Xiang Cheng's Team at the School of Electronics, Peking University
- **Related Research:** QuaDRiGa simulators, Large Language Models (LLM), channel prediction neural networks, preprocessing modules, embedding modules, pre-trained LLM modules, output modules.
- **Published Journal:** Journal of Communications and Information Networks, 2024.06
- **Paper Link:** [LLM4CP: Adapting Large Language Models for Channel Prediction](https://ieeexplore.ieee.org/document/10582829)

### **21. [The first Generative Adversarial Network model for multi-stitch embroidery](https://hyper.ai/news/34669)**

- **Research highlight:** [https://hyper.ai/news/34669](https://hyper.ai/news/34669)
- **Research Team:** Visual Computing and Digital Textile Team, School of Computer Science and AI, Wuhan Textile University
- **Related Research:** Multi-stitch embroidery datasets, Generative Adversarial Networks (GANs), CNNs, multi-stitch embroidery GAN model MSEmbGAN, region-aware texture generation networks, coloration networks. Enhances texture realism and color fidelity in embroidery.
- **Published Journal:** IEEE Transactions on Visualization and Computer Graphics, 2024
- **Paper Link:** [MSEmbGAN: Multi-Stitch Embroidery Synthesis via Region-Aware Texture Generation](https://csai.wtu.edu.cn/TVCG01/index.html)

### **22. [Fast Automated Scanning Toolkit (FAST) efficiently acquires sample information](https://hyper.ai/news/28100)**

- **Research highlight:** [https://hyper.ai/news/28100](https://hyper.ai/news/28100)
- **Research Team:** Argonne National Laboratory Research Team
- **Related Research:** SLADS-Net methods, path optimization techniques. Prioritizes heterogeneous regions and accurately replicates all major features in full-scan images.
- **Published Journal:** Nature Communications, 2023.09
- **Paper Link:** [Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy](https://www.nature.com/articles/s41467-023-40339-1)

### **23. [Population Dynamics Foundation Model PDFM open-sourced, precisely predicting US unemployment and poverty rates](https://hyper.ai/news/36380)**

- **Research highlight:** [https://hyper.ai/news/36380](https://hyper.ai/news/36380)
- **Research Team:** Google
- **Related Research:** Population Dynamics Foundation Model, predicting unemployment and poverty rates, decoupled embedding architectures, using PDFM to enhance SOTA forecasting foundation model TimesFM, aggregated search trend datasets, map datasets, busyness datasets, weather and air quality, remote sensing data, Graph Neural Networks (GNNs), enhancing existing geospatial models.
- **Published Journal:** arXiv, 2024.12
- **Paper Link:** [General Geospatial Inference with a Population Dynamics Foundation Model](https://arxiv.org/abs/2411.07207)

### **24. [Deep learning model CatGWR estimates spatial non-stationarity](https://hyper.ai/news/38055)**

- **Research highlight:** [https://hyper.ai/news/38055](https://hyper.ai/news/38055)
- **Research Team:** Zhejiang Provincial Key Laboratory of GIS
- **Related Research:** Deep learning model Context-Attention Geographically Weighted Regression, attention mechanisms, estimating spatial non-stationarity, CatGWR model, simulation experiments, preprocessing modules, zoom-in modules, regression modules.
- **Published Journal:** International Journal of Geographical Information Science, 2025.02
- **Paper Link:** [Using an attention-based architecture to incorporate context similarity into spatial non-stationarity estimation](https://doi.org/10.1080/13658816.2025.2456556)

### **25. [World's first VR exercise intervention system REVERIE reshapes youth brain-body-mind health](https://hyper.ai/news/41266)**

- **Research highlight:** [https://hyper.ai/news/41266](https://hyper.ai/news/41266)
- **Research Team:** Prof. Huating Li's Team (Shanghai Sixth People's Hospital / Institute of Active Health), Prof. Bin Sheng's Team (SJTU / MOE Key Lab of AI), Researcher Jihong Wang's Team (Shanghai University of Sport), Prof. Rong Zeng's Team (ShanghaiTech / Shanghai Clinical Research Center), Prof. Shuide Lin's Team (NUS).
- **Related Research:** Physical exercise, virtual world (metaverse) VR sports, virtual reality exercise system REVERIE, youth obesity, Transformer architectures, iterative user interactions.
- **Published Journal:** Nature Medicine, 2025.06
- **Paper Link:** [Adaptive AI-based virtual reality sports system for adolescents with excess body weight: a randomized controlled trial](https://www.nature.com/articles/s41591-025-03724-5)

### **26. [Based on over 176k inscription data, Aeneas achieves arbitrary-length restoration of ancient Roman inscriptions for the first time](https://hyper.ai/news/42141)**

- **Research highlight:** [https://hyper.ai/news/42141](https://hyper.ai/news/42141)
- **Research Team:** Google DeepMind researchers, University of Nottingham, University of Warwick, etc.
- **Related Research:** Multimodal generative neural network Aeneas, Transformer decoders, Latin inscription datasets, LED dataset, inscription restoration.
- **Published Journal:** Nature, 2025.07
- **Paper Link:** [Contextualizing ancient texts with generative neural networks](https://www.nature.com/articles/s41586-025-09292-5)

### **27. [Panoramic video generation framework PanoWan also handles zero-shot video editing](https://hyper.ai/news/42205)**

- **Research highlight:** [https://hyper.ai/news/42205](https://hyper.ai/news/42205)
- **Research Team:** Camera Intelligence Lab @ PKU (Boxin Shi's Team), OpenBayes
- **Related Research:** Panoramic video, PanoVid panoramic video dataset, zero-shot video editing, latitude-aware sampling, rotational semantic denoising, boundary-padded pixel-wise decoding.
- **Published Journal:** arXiv, 2025.06
- **Paper Link:** [PanoWan: Lifting Diffusion Video Generation Models to 360° with Latitude/Longitude-aware Mechanisms](https://arxiv.org/abs/2505.22016)

### **28. [YOLOv11-based ceramic classification intelligent framework integrates visual modeling and economic analysis, achieving artifact classification and value estimation](https://hyper.ai/news/42268)**

- **Research highlight:** [https://hyper.ai/news/42268](https://hyper.ai/news/42268)
- **Research Team:** Universiti Putra Malaysia, UNSW Sydney
- **Related Research:** Ceramic classification, CNNs, transfer learning, capsule networks, YOLOv11, ceramic image datasets, hybrid data acquisition methods, Random Forest regression models.
- **Published Journal:** Nature Partner Journals, 2025.06
- **Paper Link:** [Integrating deep learning and machine learning for ceramic artifact classification and market value prediction](https://www.nature.com/articles/s40494-025-01886-6)

### **29. ["Microwave Brain" chip born, simultaneously processing ultra-high-speed data and wireless signals with 75% accuracy at 176 milliwatts power](https://hyper.ai/news/43093)**

- **Research highlight:** [https://hyper.ai/news/43093](https://hyper.ai/news/43093)
- **Research Team:** Cornell University
- **Related Research:** High-bandwidth applications, microwave neural networks, linear regression models, RadioML2016.10A dataset, deep learning, analog computing.
- **Published Journal:** Nature Electronics, 2025.08
- **Paper Link:** [An integrated microwave neural network for broadband computation and communication](https://go.hyper.ai/rMZ2K)

### **30. [Spatiotemporal imputation and prediction model STIMP released, realizing precise predictions of coastal Chlorophyll-a distribution](https://hyper.ai/news/43613)**

- **Research highlight:** [https://hyper.ai/news/43613](https://hyper.ai/news/43613)
- **Research Team:** HKUST Research Team
- **Related Research:** Chlorophyll-a prediction, MODIS in-situ Chl-a datasets, Himawari satellite remote sensing reflectance datasets, deep learning, STIMP architecture, water body health diagnosis.
- **Published Journal:** Nature Communications, 2025.08
- **Paper Link:** [Spatiotemporal Imputation and Prediction Model](https://go.hyper.ai/BjOR5)

### **31. [MIT and others achieve high-precision prediction of plasma dynamics under few-shot conditions based on machine learning](https://hyper.ai/news/45260)**

- **Research highlight:** [https://hyper.ai/news/45260](https://hyper.ai/news/45260)
- **Research Team:** Research team led by MIT
- **Related Research:** Tokamaks, Scientific Machine Learning (SciML), Neural State-Space Models (NSSM), control-error sensitivity robustness validation, predict-first extrapolation testing.
- **Published Journal:** Nature Communications, 2025.10
- **Paper Link:** [Learning plasma dynamics and robust rampdown trajectories with predict-first experiments at TCV](https://www.nature.com/articles/s41467-025-63917-x)

### **32. [Reac-Discovery fuses mathematical modeling, machine learning, and automated experiments to solve the universality challenge of self-driving laboratory systems](https://hyper.ai/news/45626)**

- **Research highlight:** [https://hyper.ai/news/45626](https://hyper.ai/news/45626)
- **Research Team:** IMDEA Materials Institute (Spain)
- **Related Research:** Self-Driving Laboratories (SDL), Reac-Discovery semi-autonomous digital platforms, closed-loop systems integrating design/manufacturing/optimization modules, real-time NMR monitoring, ML process parameter optimization, topological descriptors, structural parameterization datasets, printability datasets, reaction performance datasets.
- **Published Journal:** Nature Communications, 2025.10
- **Paper Link:** [Reac-Discovery: an artificial intelligence–driven platform for continuous-flow catalytic reactor discovery and optimization](https://go.hyper.ai/ueB79)

### **33. [The first neuron modeling framework NOBLE validated by human cortical data is introduced](https://hyper.ai/news/45806)**

- **Research highlight:** [https://hyper.ai/news/45806](https://hyper.ai/news/45806)
- **Research Team:** ETH Zurich, Caltech, University of Alberta
- **Related Research:** Deep learning, neuron features embedding, current injection embedding, NOBLE neuron modeling framework.
- **Published Journal:** NeurIPS 2025, 2025.09
- **Paper Link:** [NOBLE – Neural Operator with Biologically-informed Latent Embeddings to Capture Experimental Variability in Biological Neuron Models](https://go.hyper.ai/Ramfp)

### **34. [Image geolocation framework LocDiff goes online, enabling grid-free and reference-library-free global precision positioning](https://hyper.ai/news/46687)**

- **Research highlight:** [https://hyper.ai/news/46687](https://hyper.ai/news/46687)
- **Research Team:** UMaine, UT Austin, UGA, UMD, Google, OpenAI, Harvard
- **Related Research:** Spherical Harmonics Dirac distributions, LocDiff ensemble framework, MP16 dataset, Im2GPS3k dataset, YFCC26k dataset, GWS15k dataset, Conditional Siren-UNet (CS-UNet) architecture, efficient computation strategies, SHDD coding schemes, image geolocation.
- **Published Journal:** NeurIPS 2025, 2025.10
- **Paper Link:** [LocDiff: Identifying Locations on Earth by Diffusing in the Hilbert Space](https://openreview.net/forum?id=ghybX0Qlls)

### **35. [Machine learning combined with py-GC-MS precisely identifies evidence of life in Archean rocks](https://hyper.ai/news/47543)**

- **Research highlight:** [https://hyper.ai/news/47543](https://hyper.ai/news/47543)
- **Research Team:** Earth and Planets Laboratory at the Carnegie Institution for Science, alongside multiple global institutions
- **Related Research:** Pyrolysis gas chromatography-mass spectrometry (py-GC-MS), supervised machine learning.
- **Published Journal:** PNAS
- **Paper Link:** [Organic geochemical evidence for life in Archean rocks identified by pyrolysis–GC–MS and supervised machine learning](https://www.pnas.org/doi/10.1073/pnas.2514534122)

### **36. [Tsinghua University team proposes neuro-symbolic regression method ND² to automatically derive complex network dynamics formulas](https://hyper.ai/news/47950)**

- **Research highlight:** [https://hyper.ai/news/47950](https://hyper.ai/news/47950)
- **Research Team:** Tsinghua University
- **Related Research:** Network dynamics, symbolic regression, ND², equation derivation, scientific machine learning.
- **Published Journal:** Nature Communications
- **Paper Link:** *(Link points to the Archean rocks paper in original Chinese, but kept numbering and reference translation as provided)*

*(Note: The provided source had a duplicate item 35 and 36 linking to PNAS Archean rocks, while the TOC indicated ND2. Translated directly based on the provided text block items 35/36)*

### **37. [Zhejiang University team proposes geologically constrained mineral prospectivity prediction method, explicitly depicting mineralization anisotropy](https://hyper.ai/news/48396)**

- **Research highlight:** [https://hyper.ai/news/48396](https://hyper.ai/news/48396)
- **Research Team:** Zhejiang University Research Team
- **Related Research:** Mineral Prospectivity Mapping (MPM), anisotropic spatial proximity neural networks, intelligent prospecting.
- **Published Journal:** Geology
- **Paper Link:** [Geologically constrained data-driven modeling for mineral prospectivity mapping](https://go.hyper.ai/vbUpa)

### **38. [Tsinghua and UChicago team publishes in Nature: AI tools expand scientists' impact but contract science's focus](https://hyper.ai/news/48748)**

- **Research highlight:** [https://hyper.ai/news/48748](https://hyper.ai/news/48748)
- **Research Team:** Joint Team from Tsinghua University and the University of Chicago
- **Related Research:** AI for Science, research productivity, scientific citation patterns, research ecosystems, scientometrics.
- **Published Journal:** Nature
- **Paper Link:** [Artificial intelligence tools expand scientists’ impact but contract science’s focus](https://www.nature.com/articles/s41586-025-09922-y)

### **39. [UC team proposes AI-augmented chip-scale spectrometer, achieving high spectral fidelity in an ultra-small volume](https://hyper.ai/news/48905)**

- **Research highlight:** [https://hyper.ai/news/48905](https://hyper.ai/news/48905)
- **Research Team:** University of California Research Team
- **Related Research:** Chip-scale spectrometers, Photon-Trapping Surface Textures (PTST), fully connected neural networks, hyperspectral imaging.
- **Published Journal:** Advanced Photonics
- **Paper Link:** [AI-augmented photon-trapping spectrometer-on-a-chip on silicon platform with extended near-infrared sensitivity](https://doi.org/10.1117/1.AP.8.1.016008)

### **40. [US DOE Oak Ridge National Lab proposes D-CHAG method, significantly reducing memory footprint for multi-channel foundation models](https://hyper.ai/news/49330)**

- **Research highlight:** [https://hyper.ai/news/49330](https://hyper.ai/news/49330)
- **Research Team:** Researchers at US DOE Oak Ridge National Laboratory
- **Related Research:** Vision science foundation models, Distributed Cross-Channel Hierarchical Aggregation (D-CHAG), Tensor Parallelism (TP), hierarchical channel aggregation.
- **Published Journal:** SC25
- **Paper Link:** [Distributed Cross-Channel Hierarchical Aggregation for Foundation Models](https://dl.acm.org/doi/10.1145/3712285.3759870)

### **41. [Polymathic AI team proposes continuum foundation model Walrus, setting records in cross-domain simulation performance](https://hyper.ai/news/49076)**

- **Research highlight:** [https://hyper.ai/news/49076](https://hyper.ai/news/49076)
- **Research Team:** Polymathic AI Collaborative Research Team
- **Related Research:** Continuum dynamics, physics simulation foundation models, Walrus model, adaptive computational tokenization.
- **Published Journal:** arXiv
- **Paper Link:** [Walrus: A Cross-Domain Foundation Model for Continuum Dynamics](https://arxiv.org/abs/2511.15684)

### **42. [EPFL proposes novel architecture DYNAMI-CAL GraphNet, a physics-informed GNN accurately modeling multi-body dynamics](https://hyper.ai/news/49808)**

- **Research highlight:** [https://hyper.ai/news/49808](https://hyper.ai/news/49808)
- **Research Team:** EPFL Research Team
- **Related Research:** Physics-informed GNN, multi-body dynamical systems, DYNAMI-CAL GraphNet, conservation of linear and angular momentum.
- **Published Journal:** Nature Communications
- **Paper Link:** [A physics-informed graph neural network conserving linear and angular momentum for dynamical systems](https://www.nature.com/articles/s41467-025-67802-5)

### **43. [MIT proposes novel method Wave-Former, achieving high-precision 3D reconstruction of completely occluded objects](https://hyper.ai/news/50018)**

- **Research highlight:** [https://hyper.ai/news/50018](https://hyper.ai/news/50018)
- **Research Team:** MIT Research Team
- **Related Research:** Computer vision, through-occlusion 3D reconstruction, mmWave sensing, Wave-Former, wireless shape completion.
- **Published Journal:** arXiv
- **Paper Link:** [Wave-Former: Through-Occlusion 3D Reconstruction via Wireless Shape Completion](https://arxiv.org/abs/2511.14152)

### **44. [MIT proposes DRiffusion draft-and-refine parallel framework, realizing lossless acceleration for diffusion model inference](https://hyper.ai/news/50209)**

- **Research highlight:** [https://hyper.ai/news/50209](https://hyper.ai/news/50209)
- **Research Team:** MIT Research Team
- **Related Research:** Diffusion Models, inference acceleration, parallelization techniques, DRiffusion, draft-and-refine.
- **Published Journal:** arXiv
- **Paper Link:** [DRiffusion: Draft-and-Refine Process Parallelizes Diffusion Models with Ease](https://arxiv.org/abs/2603.25872)

### **45. [Technion - Israel Institute of Technology proposes Task Tokens, allowing behavior foundation models to flexibly adapt to specific tasks](https://hyper.ai/news/50788)**

- **Research highlight:** [https://hyper.ai/news/50788](https://hyper.ai/news/50788)
- **Research Team:** Technion Research Team
- **Related Research:** Robotic control, imitation learning, Behavior Foundation Models (BFMs), Task Tokens, task-specific adaptation.
- **Published Conference:** ICLR 2026
- **Paper Link:** [Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models](https://hyper.ai/papers/2503.22886)

### **46. [MIT and others propose EnergAIzer framework, achieving fast and accurate GPU power estimation for AI workloads](https://hyper.ai/news/51038)**

- **Research highlight:** [https://hyper.ai/news/51038](https://hyper.ai/news/51038)
- **Research Team:** MIT and MIT-IBM Watson AI Lab
- **Related Research:** GPU power estimation, AI workloads, data center energy efficiency, EnergAIzer framework, hardware performance profiling.
- **Published Journal:** arXiv
- **Paper Link:** [EnergAIzer: Fast and Accurate GPU Power Estimation Framework for AI Workloads](https://arxiv.org/abs/2604.20105)

### **47. [UIUC proposes heterogeneous agent framework Eywa, breaking through the limits of language-centric large models](https://hyper.ai/news/51222)**

- **Research highlight:** [https://hyper.ai/news/51222](https://hyper.ai/news/51222)
- **Research Team:** UIUC Research Team
- **Related Research:** Agentic AI, heterogeneous agent framework Eywa, domain-specific foundation models, multi-agent systems, Large Language Models (LLM).
- **Published Journal:** arXiv
- **Paper Link:** [Heterogeneous Scientific Foundation Model Collaboration](https://hyper.ai/papers/2604.27351)

### **48. [Stanford University and others use LSTM surrogate models to achieve 252x accelerated simulation of second-order nonlinear optics](https://hyper.ai/news/51410)**

- **Research highlight:** [https://hyper.ai/news/51410](https://hyper.ai/news/51410)
- **Research Team:** Stanford University, UCLA, and SLAC National Accelerator Laboratory
- **Related Research:** Second-order nonlinear optics, Sum-Frequency Generation (SFG), Long Short-Term Memory networks (LSTM), Surrogate Model, Split-Step Fourier Method (SSFM).
- **Published Journal:** Advanced Photonics
- **Paper Link:** [Deep learning-assisted modeling for χ⁽²⁾ nonlinear optics](https://go.hyper.ai/5bLoA)
