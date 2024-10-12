from builtins import object

class Moe_config(object):
    def __init__(self):
        self.ffn_dim = 2048
        self.hidden_dim = 768
        self.num_experts = 5
        self.top_k = 2
        self.num_cluster = 5

    def __str__(self):
        return (f"Moe_config(ffn_dim={self.ffn_dim}, "
                f"hidden_size={self.hidden_dim}, "
                f"num_cluster={self.num_cluster}, "
                f"top_k={self.top_k})")
