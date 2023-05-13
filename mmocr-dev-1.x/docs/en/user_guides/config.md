# Config

MMOCR mainly uses Python files as configuration files. The design of its configuration file system integrates the ideas of modularity and inheritance to facilitate various experiments.

## Common Usage

```{note}
This section is recommended to be read together with the primary usage in {external+mmengine:doc}`MMEngine: Config <tutorials/config>`.
```

There are three most common operations in MMOCR: inheritance of configuration files, reference to `_base_` variables, and modification of `_base_` variables. Config provides two syntaxes for inheriting and modifying `_base_`, one for Python, Json, and Yaml, and one for Python configuration files only. In MMOCR, we **prefer the Python-only syntax**, so this will be the basis for further description.

The `configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py` is used as an example to illustrate the three common uses.

```Python
_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test = _base_.icdar2015_textdet_test
icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)
```

### Configuration Inheritance

There is an inheritance mechanism for configuration files, i.e. one configuration file A can use another configuration file B as its base and inherit all the fields directly from it, thus avoiding a lot of copy-pasting.

In `dbnet_resnet18_fpnc_1200e_icdar2015.py` you can see that

```Python
_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]
```

The above statement reads all the base configuration files in the list, and all the fields in them are loaded into `dbnet_resnet18_fpnc_1200e_icdar2015.py`. We can see the structure of the configuration file after it has been parsed by running the following statement in a Python interpretation.

```Python
from mmengine import Config
db_config = Config.fromfile('configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py')
print(db_config)
```

It can be found that the parsed configuration contains all the fields and information in the base configuration.

```{note}
Variables with the same name cannot exist in each `_base_` profile.
```

### `_base_` Variable References

Sometimes we may need to reference some fields in the `_base_` configuration directly in order to avoid duplicate definitions. Suppose we want to get the variable `pseudo` in the `_base_` configuration, we can get the variable in the `_base_` configuration directly via `_base_.pseudo`.

This syntax has been used extensively in the configuration of MMOCR, and the dataset and pipeline configurations for each model in MMOCR are referenced in the *_base_* configuration. For example,

```Python
icdar2015_textdet_train = _base_.icdar2015_textdet_train
# ...
train_dataloader = dict(
    # ...
    dataset=icdar2015_textdet_train)
```

<div id="base_variable_modification"></div>

### `_base_` Variable Modification

In MMOCR, different algorithms usually have different pipelines in different datasets, so there are often scenarios to modify the `pipeline` in the dataset. There are also many scenarios where you need to modify variables in the `_base_` configuration, for example, modifying the training strategy of an algorithm, replacing some modules of an algorithm(backbone, etc.). Users can directly modify the referenced `_base_` variables using Python syntax. For dict, we also provide a method similar to class attribute modification to modify the contents of the dictionary directly.

1. Dictionary

   Here is an example of modifying `pipeline` in a dataset.

   The dictionary can be modified using Python syntax:

   ```Python
   # Get the dataset in _base_
   icdar2015_textdet_train = _base_.icdar2015_textdet_train
   # You can modify the variables directly with Python's update
   icdar2015_textdet_train.update(pipeline=_base_.train_pipeline)
   ```

   It can also be modified in the same way as changing Python class attributes.

   ```Python
   # Get the dataset in _base_
   icdar2015_textdet_train = _base_.icdar2015_textdet_train
   # The class property method is modified
   icdar2015_textdet_train.pipeline = _base_.train_pipeline
   ```

2. List

   Suppose the variable `pseudo = [1, 2, 3]` in the `_base_` configuration needs to be modified to `[1, 2, 4]`:

   ```Python
   # pseudo.py
   pseudo = [1, 2, 3]
   ```

   Can be rewritten directly as.

   ```Python
   _base_ = ['pseudo.py']
   pseudo = [1, 2, 4]
   ```

   Or modify the list using Python syntax:

   ```Python
   _base_ = ['pseudo.py']
   pseudo = _base_.pseudo
   pseudo[2] = 4
   ```

### Command Line Modification

Sometimes we only want to fix part of the configuration and do not want to modify the configuration file itself. For example, if you want to change the learning rate during an experiment but do not want to write a new configuration file, you can pass in parameters on the command line to override the relevant configuration.

We can pass `--cfg-options` on the command line and modify the corresponding fields directly with the arguments after it. For example, we can run the following command to modify the learning rate temporarily for this training session.

```Shell
python tools/train.py example.py --cfg-options optim_wrapper.optimizer.lr=1
```

For more detailed usage, refer to {external+mmengine:doc}`MMEngine: Command Line Modification <tutorials/config>`.

## Configuration Content

With config files and Registry, MMOCR can modify the training parameters as well as the model configuration without invading the code. Specifically, users can customize the following modules in the configuration file: environment configuration, hook configuration, log configuration, training strategy configuration, data-related configuration, model-related configuration, evaluation configuration, and visualization configuration.

This document will take the text detection algorithm `DBNet` and the text recognition algorithm `CRNN` as examples to introduce the contents of Config in detail.

<div id="env_config"></div>

### Environment Configuration

```Python
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
```

There are three main components:

- Set the default `scope` of all registries to `mmocr`, ensuring that all modules are searched first from the `MMOCR` codebase. If the module does not exist, the search will continue from the upstream algorithm libraries `MMEngine` and `MMCV`, see {external+mmengine:doc}`MMEngine: Registry <tutorials/registry>` for more details.

- `env_cfg` configures the distributed environment, see {external+mmengine:doc}`MMEngine: Runner <tutorials/runner>` for more details.

- `randomness`: Some settings to make the experiment as reproducible
  as possible like seed and deterministic. See {external+mmengine:doc}`MMEngine: Runner <tutorials/runner>` for more details.

<div id="hook_config"></div>

### Hook Configuration

Hooks are divided into two main parts, default hooks, which are required for all tasks to run, and custom hooks, which generally serve specific algorithms or specific tasks (there are no custom hooks in MMOCR so far).

```Python
default_hooks = dict(
    timer=dict(type='IterTimerHook'), # Time recording, including data time as well as model inference time
    logger=dict(type='LoggerHook', interval=1), # Collect logs from different components
    param_scheduler=dict(type='ParamSchedulerHook'), # Update some hyper-parameters in optimizer
    checkpoint=dict(type='CheckpointHook', interval=1),# Save checkpoint. `interval` control save interval
    sampler_seed=dict(type='DistSamplerSeedHook'), # Data-loading sampler for distributed training.
    sync_buffer=dict(type='SyncBuffersHook'), # Synchronize buffer in case of distributed training
    visualization=dict( # Visualize the results of val and test
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
 custom_hooks = []
```

Here is a brief description of a few hooks whose parameters may be changed frequently. For a general modification method, refer to <a href="#base_variable_modification">Modify configuration</a>.

- `LoggerHook`: Used to configure the behavior of the logger. For example, by modifying `interval` you can control the interval of log printing, so that the log is printed once per `interval` iteration, for more settings refer to [LoggerHook API](mmengine.hooks.LoggerHook).

- `CheckpointHook`: Used to configure checkpoint-related behavior, such as saving optimal and/or latest weights. You can also modify `interval` to control the checkpoint saving interval. More settings can be found in [CheckpointHook API](mmengine.hooks.CheckpointHook)

- `VisualizationHook`: Used to configure visualization-related behavior, such as visualizing predicted results during validation or testing. **Default is off**. This Hook also depends on [Visualization Configuration](#Visualization-configuration). You can refer to [Visualizer](visualization.md) for more details. For more configuration, you can refer to [VisualizationHook API](mmocr.engine.hooks.VisualizationHook).

If you want to learn more about the configuration of the default hooks and their functions, you can refer to {external+mmengine:doc}`MMEngine: Hooks <tutorials/hook>`.

<div id="log_config"></div>

### Log Configuration

This section is mainly used to configure the log level and the log processor.

```Python
log_level = 'INFO' # Logging Level
log_processor = dict(type='LogProcessor',
                        window_size=10,
                        by_epoch=True)
```

- The logging severity level is the same as that of {external+python:doc}`Python: logging <library/logging>`

- The log processor is mainly used to control the format of the output, detailed functions can be found in {external+mmengine:doc}`MMEngine: logging <advanced_tutorials/logging>`.

  - `by_epoch=True` indicates that the logs are output in accordance to "epoch", and the log format needs to be consistent with the `type='EpochBasedTrainLoop'` parameter in `train_cfg`. For example, if you want to output logs by iteration number, you need to set ` by_epoch=False` in `log_processor` and `type='IterBasedTrainLoop'` in `train_cfg`.

  - `window_size` indicates the smoothing window of the loss, i.e. the average value of the various losses for the last `window_size` iterations. the final loss value printed in logger is the average of all the losses.

  <div id="schedule_config"></div>

### Training Strategy Configuration

This section mainly contains optimizer settings, learning rate schedules and `Loop` settings.

Training strategies usually vary for different tasks (text detection, text recognition, key information extraction). Here we explain the example configuration in `CRNN`, which is a text recognition model.

```Python
# optimizer
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adadelta', lr=1.0))
param_scheduler = [dict(type='ConstantLR', factor=1.0)]
train_cfg = dict(type='EpochBasedTrainLoop',
                    max_epochs=5, # train epochs
                    val_interval=1) # val interval
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

- `optim_wrapper` : It contains two main parts, OptimWrapper and Optimizer. Detailed usage information can be found in {external+mmengine:doc}`MMEngine: Optimizer Wrapper <tutorials/optim_wrapper>`.

  - The Optimizer wrapper supports different training strategies, including mixed-accuracy training (AMP), gradient accumulation, and gradient truncation.

  - All PyTorch optimizers are supported in the optimizer settings. All supported optimizers are available in {external+torch:ref}`PyTorch Optimizer List <optim:algorithms>`.

- `param_scheduler` : learning rate tuning strategy, supports most of the learning rate schedulers in PyTorch, such as `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR`, etc., and is used in much the same way, see [scheduler interface](mmengine.optim.scheduler), and more features can be found in the {external+mmengine:doc}`MMEngine: Optimizer Parameter Tuning Strategy <tutorials/param_scheduler>`.

- `train/test/val_cfg` : the execution flow of the task, MMEngine provides four kinds of flow: `EpochBasedTrainLoop`, `IterBasedTrainLoop`, `ValLoop`, `TestLoop` More can be found in {external+mmengine:doc}`MMEngine: loop controller <design/runner>`.

### Data-related Configuration

<div id="dataset_config"></id>

#### Dataset Configuration

It is mainly about two parts.

- The location of the dataset(s), including images and annotation files.

- Data augmentation related configurations. In the OCR domain, data augmentation is usually strongly associated with the model.

More parameter configurations can be found in [Data Base Class](#TODO).

The naming convention for dataset fields in MMOCR is

```Python
{dataset}_{task}_{train/val/test} = dict(...)
```

- dataset: See [dataset abbreviations](#TODO)

- task: `det`(text detection), `rec`(text recognition), `kie`(key information extraction)

- train/val/test: Dataset split.

For example, for text recognition tasks, Syn90k is used as the training set, while icdar2013 and icdar2015 serve as the test sets. These are configured as follows.

```Python
# text recognition dataset configuration
mjsynth_textrecog_train = dict(
    type='OCRDataset',
    data_root='data/rec/Syn90k/',
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

icdar2013_textrecog_test = dict(
    type='OCRDataset',
    data_root='data/rec/icdar_2013/',
    data_prefix=dict(img_path='Challenge2_Test_Task3_Images/'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)

icdar2015_textrecog_test = dict(
    type='OCRDataset',
    data_root='data/rec/icdar_2015/',
    data_prefix=dict(img_path='ch4_test_word_images_gt/'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
```

<div id="pipeline_config"></div>

#### Data Pipeline Configuration

In MMOCR, dataset construction and data preparation are decoupled from each other. In other words, dataset classes such as `OCRDataset` are responsible for reading and parsing annotation files, while Data Transforms further implement data loading, data augmentation, data formatting and other related functions.

In general, there are different augmentation strategies for training and testing, so there are usually `training_pipeline` and `testing_pipeline`. More information can be found in [Data Transforms](../basic_concepts/transforms.md)

- The data augmentation process of the training pipeline is usually: data loading (LoadImageFromFile) -> annotation information loading (LoadXXXAnntation) -> data augmentation -> data formatting (PackXXXInputs).

- The data augmentation flow of the test pipeline is usually: Data Loading (LoadImageFromFile) -> Data Augmentation -> Annotation Loading (LoadXXXAnntation) -> Data Formatting (PackXXXInputs).

Due to the specificity of the OCR task, different models have different data augmentation techniques, and even the same model can have different data augmentation strategies for different datasets. Take `CRNN` as an example.

```Python
# Data Augmentation
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        ignore_empty=True,
        min_size=5),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(100, 32), keep_ratio=False),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale'),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=None,
        width_divisor=16),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
```

#### Dataloader Configuration

The main configuration information needed to construct the dataset loader (dataloader), see {external+torch:doc}`PyTorch DataLoader <data>` for more tutorials.

```Python
# Dataloader
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[mjsynth_textrecog_train],
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[icdar2013_textrecog_test, icdar2015_textrecog_test],
        pipeline=test_pipeline))
test_dataloader = val_dataloader
```

### Model-related Configuration

<div id="model_config"></div>

#### Network Configuration

This section configures the network architecture. Different algorithmic tasks use different network architectures. Find more info about network architecture in [structures](../basic_concepts/structures.md)

##### Text Detection

Text detection consists of several parts:

- `data_preprocessor`: [data_preprocessor](mmocr.models.textdet.data_preprocessors.TextDetDataPreprocessor)
- `backbone`: backbone network configuration
- `neck`: neck network configuration
- `det_head`: detection head network configuration
  - `module_loss`: module loss configuration
  - `postprocessor`: postprocessor configuration

We present the model configuration in text detection using DBNet as an example.

```Python
model = dict(
    type='DBNet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32)
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe'),
    neck=dict(
        type='FPNC', in_channels=[64, 128, 256, 512], lateral_channels=256),
    det_head=dict(
        type='DBHead',
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')))
```

##### Text Recognition

Text recognition mainly contains:

- `data_processor`: [data preprocessor configuration](mmocr.models.textrecog.data_processors.TextRecDataPreprocessor)
- `preprocessor`: network preprocessor configuration, e.g. TPS
- `backbone`: backbone configuration
- `encoder`: encoder configuration
- `decoder`: decoder configuration
  - `module_loss`: decoder module loss configuration
  - `postprocessor`: decoder postprocessor configuration
  - `dictionary`: dictionary configuration

Using CRNN as an example.

```Python
# model
model = dict(
   type='CRNN',
   data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127], std=[127])
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(
        type='CRNNDecoder',
        in_channels=512,
        rnn_flag=True,
        module_loss=dict(type='CTCModuleLoss', letter_case='lower'),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dict(
            type='Dictionary',
            dict_file='dicts/lower_english_digits.txt',
            with_padding=True)))
```

<div id="weight_config"></div>

#### Checkpoint Loading Configuration

The model weights in the checkpoint file can be loaded via the `load_from` parameter, simply by setting the `load_from` parameter to the path of the checkpoint file.

You can also resume training by setting `resume=True` to load the training status information in the checkpoint. When both `load_from` and `resume=True` are set, MMEngine will load the training state from the checkpoint file at the `load_from` path.

If only `resume=True` is set, the executor will try to find and read the latest checkpoint file from the `work_dir` folder

```Python
load_from = None # Path to load checkpoint
resume = False # whether resume
```

More can be found in {external+mmengine:doc}`MMEngine: Load Weights or Recover Training <tutorials/runner>` and [OCR Advanced Tips - Resume Training from Checkpoints](train_test.md#resume-training-from-a-checkpoint).

<div id="eval_config"></id>

### Evaluation Configuration

In model validation and model testing, quantitative measurement of model accuracy is often required. MMOCR performs this function by means of `Metric` and `Evaluator`. For more information, please refer to {external+mmengine:doc}`MMEngine: Evaluation <tutorials/evaluation>` and [Evaluation](../basic_concepts/evaluation.md)

#### Evaluator

Evaluator is mainly used to manage multiple datasets and multiple `Metrics`. For single and multiple dataset cases, there are single and multiple dataset evaluators, both of which can manage multiple `Metrics`.

The single-dataset evaluator is configured as follows.

```Python
# Single Dataset Single Metric
val_evaluator = dict(
    type='Evaluator',
    metrics=dict())

# Single Dataset Multiple Metric
val_evaluator = dict(
    type='Evaluator',
    metrics=[...])
```

`MultiDatasetsEvaluator` differs from single-dataset evaluation in two aspects: `type` and `dataset_prefixes`. The evaluator type must be `MultiDatasetsEvaluator` and cannot be omitted. The `dataset_prefixes` is mainly used to distinguish the results of different datasets with the same evaluation metrics, see [MultiDatasetsEvaluation](../basic_concepts/evaluation.md).

Assuming that we need to test accuracy on IC13 and IC15 datasets, the configuration is as follows.

```Python
#  Multiple datasets, single Metric
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=dict(),
    dataset_prefixes=['IC13', 'IC15'])

# Multiple datasets, multiple Metrics
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[...],
    dataset_prefixes=['IC13', 'IC15'])
```

#### Metric

A metric evaluates a model's performance from a specific perspective. While there is no such common metric that fits all the tasks, MMOCR provides enough flexibility such that multiple metrics serving the same task can be used simultaneously. Here we list task-specific metrics for reference.

Text detection: [`HmeanIOUMetric`](mmocr.evaluation.metrics.HmeanIOUMetric)

Text recognition: [`WordMetric`](mmocr.evaluation.metrics.WordMetric), [`CharMetric`](mmocr.evaluation.metrics.CharMetric), [`OneMinusNEDMetric`](mmocr.evaluation.metrics.OneMinusNEDMetric)

Key information extraction: [`F1Metric`](mmocr.evaluation.metrics.F1Metric)

Text detection as an example, using a single `Metric` in the case of single dataset evaluation.

```Python
val_evaluator = dict(type='HmeanIOUMetric')
```

Take text recognition as an example, multiple datasets (`IC13` and `IC15`) are evaluated using multiple `Metric`s (`WordMetric` and `CharMetric`).

```Python
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ],
    dataset_prefixes=['IC13', 'IC15'])
test_evaluator = val_evaluator
```

<div id="vis_config"></div>

### Visualization Configuration

Each task is bound to a task-specific visualizer. The visualizer is mainly used for visualizing or storing intermediate results of user models and visualizing val and test prediction results. The visualization results can also be stored in different backends such as WandB, TensorBoard, etc. through the corresponding visualization backend. Commonly used modification operations can be found in [visualization](visualization.md).

The default configuration of visualization for text detection is as follows.

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',  # Different visualizers for different tasks
    vis_backends=vis_backends,
    name='visualizer')
```

## Directory Structure

All configuration files of `MMOCR` are placed under the `configs` folder. To avoid config files from being too long and improve their reusability and clarity, MMOCR takes advantage of the inheritance mechanism and split config files into eight sections. Since each section is closely related to the task type, MMOCR provides a task folder for each task in `configs/`, namely `textdet` (text detection task), `textrecog` (text recognition task), and `kie` (key information extraction). Each folder is further divided into two parts: `_base_` folder and algorithm configuration folders.

1. the `_base_` folder stores some general config files unrelated to specific algorithms, and each section is divided into datasets, training strategies and runtime configurations by directory.

2. The algorithm configuration folder stores config files that are strongly related to the algorithm. The algorithm configuration folder has two kinds of config files.

   1. Config files starting with `_base_`: Configures the model and data pipeline of an algorithm. In OCR domain, data augmentation strategies are generally strongly related to the algorithm, so the model and data pipeline are usually placed in the same config file.

   2. Other config files, i.e. the algorithm-specific configurations on the specific dataset(s): These are the full config files that further configure training and testing settings, aggregating `_base_` configurations that are scattered in different locations. Inside some modifications to the fields in `_base_` configs may be performed, such as data pipeline, training strategy, etc.

All these config files are distributed in different folders according to their contents as follows:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>

<table class="tg">
<thead>
  <tr>
    <td class="tg-9wq8" rowspan="5">textdet<br></td>
    <td class="tg-lboi" rowspan="3">_base_</td>
    <td class="tg-9wq8">datasets</td>
    <td class="tg-0pky">icdar_datasets.py<br>ctw1500.py<br>...</td>
    <td class="tg-0pky"><a href="#dataset_config">Dataset configuration</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">schedules</td>
    <td class="tg-0pky">schedule_adam_600e.py<br>...</td>
    <td class="tg-0pky"><a href="#schedule_config">Training Strategy Configuration</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">default_runtime.py<br></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"><a href="#env_config">Environment Configuration</a><br><a href="#hook_config">Hook Configuration</a><br><a href="#log_config">Log Configuration</a> <br><a href="#weight_config">Checkpoint Loading Configuration</a> <br><a href="#eval_config">Evaluation Configuration</a> <br><a href="#vis_config">Visualization Configuration</a></td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">dbnet</td>
    <td class="tg-9wq8">_base_dbnet_resnet18_fpnc.py</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">-</span></td>
    <td class="tg-0pky"><a href="#model_config">Network Configuration</a> <br><a href="#pipeline_config">Data Pipeline Configuration</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">dbnet_resnet18_fpnc_1200e_icdar2015.py</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">-</span></td>
    <td class="tg-0pky"><a href="#dataloader_config">Dataloader Configuration</a> <br><a href="#pipeline_config">Data Pipeline Configuration(Optional)</a></td>
  </tr>
</thead>
</table>

The final directory structure is as follows.

```Python
configs
├── textdet
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── icdar2015.py
│   │   │   ├── icdar2017.py
│   │   │   └── totaltext.py
│   │   ├── schedules
│   │   │   └── schedule_adam_600e.py
│   │   └── default_runtime.py
│   └── dbnet
│       ├── _base_dbnet_resnet18_fpnc.py
│       └── dbnet_resnet18_fpnc_1200e_icdar2015.py
├── textrecog
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── icdar2015.py
│   │   │   ├── icdar2017.py
│   │   │   └── totaltext.py
│   │   ├── schedules
│   │   │   └── schedule_adam_base.py
│   │   └── default_runtime.py
│   └── crnn
│       ├── _base_crnn_mini-vgg.py
│       └── crnn_mini-vgg_5e_mj.py
└── kie
    ├── _base_
    │   ├──datasets
    │   └── default_runtime.py
    └── sgdmr
        └── sdmgr_novisual_60e_wildreceipt_openset.py
```

## Naming Conventions

MMOCR has a convention to name config files, and contributors to the code base need to follow the same naming rules. The file names are divided into four sections: algorithm information, module information, training information, and data information. Words that logically belong to different sections are connected by an underscore `'_'`, and multiple words in the same section are connected by a hyphen `'-'`.

```Python
{{algorithm info}}_{{module info}}_{{training info}}_{{data info}}.py
```

- algorithm info: the name of the algorithm, such as dbnet, crnn, etc.

- module info: list some intermediate modules in the order of data flow. Its content depends on the algorithm, and some modules strongly related to the model will be omitted to avoid an overly long name. For example:

  - For the text detection task and the key information extraction task :

    ```Python
    {{algorithm info}}_{{backbone}}_{{neck}}_{{head}}_{{training info}}_{{data info}}.py
    ```

    `{head}` is usually omitted since it's algorithm-specific.

  - For text recognition tasks.

    ```Python
    {{algorithm info}}_{{backbone}}_{{encoder}}_{{decoder}}_{{training info}}_{{data info}}.py
    ```

    Since encoder and decoder are generally bound to the algorithm, they are usually omitted.

- training info: some settings of the training strategy, including batch size, schedule, etc.

- data info: dataset name, modality, input size, etc., such as icdar2015 and synthtext.
