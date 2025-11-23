import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim


class VAE(nn.Module):
  """
  PyTorch model for a VAE (c_dim = 0) or Conditional VAE (c_dim > 0)

  (attrs, acts) -> (e_attrs, e_acts) -> (e_attrs, cf) -> t -> z -> t_rec -> (e_attrs_rec, cf_rec) -> (e_attrs_rec, e_acts_rec) -> (attrs_rec, acts_rec)
  """
  def __init__(self, encoder, decoder, trace_attributes=[], num_activities=12, num_resources=5, max_trace_length=10,
               num_lstm_layers=1, attr_e_dim=4, act_e_dim=3, res_e_dim=3, cf_dim=5, z_dim=20, c_dim=0,
               dropout_p=0.1, device='cpu', pad_activity_idx=0, pad_resource_idx=0):
    super(VAE, self).__init__()

    self.z_dim = z_dim
    self.is_conditional = False if c_dim == 0 else True

    # encoder
    self.encoder = encoder(
      trace_attributes=trace_attributes,
      num_activities=num_activities,
      num_resources=num_resources,
      num_lstm_layers=num_lstm_layers,
      attr_e_dim=attr_e_dim,
      act_e_dim=act_e_dim,
      res_e_dim=res_e_dim,
      cf_dim=cf_dim,
      z_dim=z_dim,
      c_dim=c_dim,
      dropout_p=dropout_p,
      is_conditional=self.is_conditional,
      pad_activity_idx=pad_activity_idx,
      pad_resource_idx=pad_resource_idx,
    )

    # decoder
    self.decoder = decoder(
      trace_attributes=trace_attributes,
      max_trace_length=max_trace_length,
      num_activities=num_activities,
      num_resources=num_resources,
      num_lstm_layers=num_lstm_layers,
      act_e_dim=act_e_dim,
      res_e_dim=res_e_dim,
      cf_dim=cf_dim,
      t_dim=self.encoder.t_dim,
      z_dim=z_dim,
      c_dim=c_dim,
      dropout_p=dropout_p,
      tot_attr_e_dim=self.encoder.tot_attr_e_dim,
      is_conditional=self.is_conditional,
      encoder_act2e=self.encoder.act2e,
      encoder_res2e=self.encoder.res2e,
      device=device,
    )

  # p(z|x)
  def encode(self, x, c=None):
    return self.encoder(x, c)

  # p(x|z)
  def decode(self, z, c=None,  **kwargs):
    """Decode latent samples.

    BEAM SEARCH CHANGE:
      Added passthrough **kwargs so that downstream decoder (e.g.,
      TransformerFlowDecoderSimplex) can accept optional arguments like:
        - use_beam_search: bool (default False)
        - beam_size: int (default 1)
        - beam_max_len: optional override for maximum generated length
      Existing callers remain unaffected because kwargs are optional.
    """
    return self.decoder(z, c)

  def forward(self, x, c=None, tuple_inputs=None, tf_ratio=0):
    mean, var = self.encode(x, c)
    epsilon = torch.randn_like(var)
    z = mean + var*epsilon # reparametrization trick
    
    return self.decode(z, c), mean, var #  add x[1:] for flow decoder
  
  # def forward(self, x, c=None, tuple_inputs=None, tf_ratio=0):
  #   mean, logvar = self.encode(x, c)  # ← rename to clarify it's log-variance
  #   std = torch.exp(0.5 * logvar)     # ← convert log-variance to standard deviation
  #   epsilon = torch.randn_like(std)
  #   z = mean + std*epsilon            # ← proper reparameterization trick
    
  #   return self.decode(z, c), mean, logvar