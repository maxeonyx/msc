import transformers
import transformers.models.deberta.modeling_tf_deberta as deberta

from ml import models

class Deberta(models.SequenceModelBase):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, is_recurrent=False, **kwargs)
        
        self.encoder = deberta.TFDebertaEncoder()
