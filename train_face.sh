export MODEL_NAME="./sd-models/model.safetensors"
export INSTANCE_DIR="./train/aki"
export OUTPUT_DIR="/content/drive/Shareddrives/milo01_h/sd/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora"

# Train related params | 训练相关参数
resolution="768"  # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
batch_size=1          # batch size
max_train_epoches=10  # max train epoches | 最大训练 epoch
save_every_n_epochs=1 # save every n epochs | 每 N 个 epoch 保存一次

network_dim=64   # network dim | 常用 4~128，不是越大越好
network_alpha=64 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

train_unet_only=0         # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
train_text_encoder_only=0 # train Text Encoder only | 仅训练 文本编码器

noise_offset=0.1 # noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为0.1
keep_tokens=1  # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。

# Learning rate | 学习率
lr="1"
unet_lr=$lr
text_encoder_lr="0.5"
lr_scheduler="cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
lr_warmup_steps=0                   # warmup steps | 仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
lr_restart_cycles=3                 # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。

# Output settings | 输出设置
output_name="maouii_wf_v4_bs1"           # output model name | 模型保存名称
save_model_as="safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors

# 其他设置
network_weights=""               # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
min_bucket_reso=256              # arb min resolution | arb 最小分辨率
max_bucket_reso=768             # arb max resolution | arb 最大分辨率
persistent_data_loader_workers=1 # persistent dataloader workers | 容易爆内存，保留加载训练集的worker，减少每个 epoch 之间的停顿
clip_skip=1                      # clip skip | 玄学 一般用 2

# 优化器设置
use_8bit_adam=0 # use 8bit adam optimizer | 使用 8bit adam 优化器节省显存，默认启用。部分 10 系老显卡无法使用，修改为 0 禁用。
use_lion=0      # use lion optimizer | 使用 Lion 优化器
use_dadaptation=1 # 使用dadaptation预训练获得峰值学习率
# LoCon 训练设置
enable_locon_train=0 # enable LoCon train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
conv_dim=4           # conv dim | 类似于 network_dim，推荐为 4
conv_alpha=4         # conv alpha | 类似于 network_alpha，可以采用与 conv_dim 一致或者更小的值

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

network_module="networks.lora"
extArgs=()

if [ $train_unet_only == 1 ]; then extArgs+=("--network_train_unet_only"); fi

if [ $train_text_encoder_only == 1 ]; then extArgs+=("--network_train_text_encoder_only"); fi

if [ $network_weights ]; then extArgs+=("--network_weights $network_weights"); fi

if [ $reg_data_dir ]; then extArgs+=("--reg_data_dir $reg_data_dir"); fi

if [ $use_8bit_adam == 1 ]; then extArgs+=("--use_8bit_adam"); fi

if [ $use_lion == 1 ]; then extArgs+=("--use_lion_optimizer"); fi
if [ $use_dadaptation == 1 ]; then extArgs+=("--optimizer_type DAdaptation");extArgs+=("--optimizer_args decouple=True"); fi
if [ $persistent_data_loader_workers == 1 ]; then extArgs+=("--persistent_data_loader_workers"); fi

if [ $enable_locon_train == 1 ]; then
  network_module="locon.locon_kohya"
  extArgs+=("--network_args conv_dim=$conv_dim conv_alpha=$conv_alpha")
fi

if [ $noise_offset ]; then extArgs+=("--noise_offset $noise_offset"); fi
  
  
lora_pti \
  --reg_data_dir="./train/reg" \
  --keep_tokens=$keep_tokens \
  --max_token_length=225 \
  --prior_loss_weight=1 \
  --clip_skip=$clip_skip \
  --cache_latents \
  --seed="3247" \
  --mixed_precision="fp16" \
  --enable_bucket \
  --logging_dir="./logs" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=$resolution \
  --train_batch_size=$batch_size \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --scale_lr \
  --learning_rate_unet=2e-4 \
  --learning_rate_text=1e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=$lr_warmup_steps \
  --lr_scheduler_lora=$lr_scheduler \
  --lr_warmup_steps_lora=100 \
  --placeholder_tokens="<s1>|<s2>" \
  --placeholder_token_at_data="<krk>|<s1><s2>" \
  --save_steps=100 \
  --max_train_steps_ti=500 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.000 \
  --device="cuda:0" \
  --lora_rank=8 \
  --use_face_segmentation_condition \
  --lora_dropout_p=0.1 \
  --lora_scale=8.0 \
  --xformers --shuffle_caption ${extArgs[@]}
