from transformers import AutoConfig


llama_config = AutoConfig.for_model(
    # llama3 config
    model_type                  = "llama",
    attention_bias              =  False,
    attention_dropout           =  0.0,
    bos_token_id                = 128000,
    eos_token_id                = 128001,
    hidden_act                  = "silu",
    hidden_size                 = 4096,
    initializer_range           = 0.02,
    #  intermediate_size 14336
    intermediate_size           = 14336,
    max_position_embeddings     = 8192,
    num_attention_heads         = 32,
    num_hidden_layers           = 32,
    num_key_value_heads         = 8,
    pretraining_tp              = 1,
    rms_norm_eps                = 1e-05,
    rope_scaling                = None,
    rope_theta                  = 500000.0,
    tie_word_embeddings         = False,
    torch_dtype                 = "bfloat16",
    transformers_version        = "4.40.0.dev0",
    use_cache                   = True,
    vocab_size                  = 128256,
    num_cluster                 = 4, 
)
