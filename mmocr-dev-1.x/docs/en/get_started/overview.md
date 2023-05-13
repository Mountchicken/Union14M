# Overview

MMOCR is an open source toolkit based on [PyTorch](https://pytorch.org/) and [MMDetection](https://github.com/open-mmlab/mmdetection), supporting numerous OCR-related models, including text detection, text recognition, and key information extraction. In addition, it supports widely-used academic datasets and provides many useful tools, assisting users in exploring various aspects of models and datasets and implementing high-quality algorithms. Generally, it has the following features.

- **One-stop, Multi-model**: MMOCR supports various OCR-related tasks and implements the latest models for text detection, recognition, and key information extraction.
- **Modular Design**: MMOCR's modular design allows users to define and reuse modules in the model on demand.
- **Various Useful Tools**: MMOCR provides a number of analysis tools, including visualizers, validation scripts, evaluators, etc., to help users troubleshoot, finetune or compare models.
- **Powered by [OpenMMLab](https://openmmlab.com/)**: Like other algorithm libraries in OpenMMLab family, MMOCR follows OpenMMLab's rigorous development guidelines and interface conventions, significantly reducing the learning cost of users familiar with other projects in OpenMMLab family. In addition, benefiting from the unified interfaces among OpenMMLab, you can easily call the models implemented in other OpenMMLab projects (e.g. MMDetection) in MMOCR, facilitating cross-domain research and real-world applications.

Together with the release of OpenMMLab 2.0, MMOCR now also comes to its 1.0.0 version, which has made significant BC-breaking changes, resulting in less code redundancy, higher code efficiency and an overall more systematic and consistent design.

Considering that there are some backward incompatible changes in this version compared to 0.x, we have prepared a detailed [migration guide](../migration/overview.md). It lists all the changes made in the new version and the steps required to migrate. We hope this guide can help users familiar with the old framework to complete the upgrade as quickly as possible. Though this may take some time, we believe that the new features brought by MMOCR and the OpenMMLab ecosystem will make it all worthwhile. 😊

Next, please read the section according to your actual needs.

- We recommend that beginners go through [Quick Run](quick_run.md) to get familiar with MMOCR and master the usage of MMOCR by reading the examples in **User Guides**.
- Intermediate and advanced developers are suggested to learn the background, conventions, and recommended implementations of each component from **Basic Concepts**.
- Read our [FAQ](faq.md) to find answers to frequently asked questions.
- If you can't find the answers you need in the documentation, feel free to raise an [issue](https://github.com/open-mmlab/mmocr/issues).
- Everyone is welcome to be a contributor! Read the [contribution guide](../notes/contribution_guide.md) to learn how to contribute to MMOCR!
