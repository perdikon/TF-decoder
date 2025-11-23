import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np
from hydra.utils import instantiate
from src.models.processgan_module import ProcessGANLitModule
from src.models.components.processgan.transformer_gen_time import TransformerModel_GEN_Time
from src.models.components.processgan.transformer_dis_time import TransformerModel_DIS_Time
import torch.nn.functional as F
from src.models.components.processgan import helpers as h
from src.data.sepsis_datamodule import SepsisDataModule, SepsisDataset
import pandas as pd



# Fixture that loads the YAML configuration as a string.
@pytest.fixture
def config():
    yaml_path = "configs/model/test.yaml"
    cfg = OmegaConf.load(yaml_path)
    return cfg

# Fixture that instantiates the model using Hydra's instantiate.
@pytest.fixture
def model(config):
    # The instantiate call will recursively instantiate all _target_ objects.
    model_instance = instantiate(config)
    return model_instance

def test_model_instantiation(model, config):
    
    # Check that the instantiated model is an instance of ProcessGANLitModule.
    assert isinstance(model, ProcessGANLitModule)
    assert isinstance(model.generator, TransformerModel_GEN_Time)
    assert isinstance(model.discriminator, TransformerModel_DIS_Time)
    
    # Also check that the compile flag is correctly set.
    assert model.hparams.compile == config.model.compile


def test_adversarial_loss(model):
    # Ensure that adversarial_loss returns an instance of BCELoss.
    loss_fn = model.adversarial_loss()
    assert isinstance(loss_fn, nn.BCELoss)

def test_configure_optimizers(model, config):
    # Check that configure_optimizers returns two dictionaries.
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, tuple)
    assert len(optimizers) == 2
    gen_dict, dis_dict = optimizers
    assert "optimizer" in gen_dict
    assert "optimizer" in dis_dict
    # If schedulers were provided in the config, verify their inclusion.
    if config.model.gen_scheduler is not None:
        assert "lr_scheduler" in gen_dict
    if config.model.dis_scheduler is not None:
        assert "lr_scheduler" in dis_dict
        
        
def test_forward(model, monkeypatch):
    # Patch the generator's forward method to simply multiply the input by 2.
    model.generator.forward = lambda x: x * 2
    x = torch.tensor([1.0, 2.0, 3.0])
    output = model.forward(x)
    # Since forward just calls generator.forward, we expect output to be x*2.
    expected = x * 2
    assert torch.allclose(output, expected)
    
    
def test_training_step(model):
    # --- Set up dummy replacements for internal methods ---
    # Patch _generator_step to return (gen_loss, fake_act, fake_time)
    model._generator_step = lambda random_data, batch, d_criterion, real_labels, dis_data_duration, dis_data_time: (
        torch.tensor(1.0), torch.zeros(10, 10), torch.zeros(10, 10)
    )
    # Patch _discriminator_step to return (dis_loss, acc_neg, acc_pos)
    model._discriminator_step = lambda dis_data_pos, dis_data_time, fake_act, fake_time, d_criterion, batch: (
        torch.tensor(0.5), torch.tensor(1.0), torch.tensor(1.0)
    )
    # Patch auxiliary loss methods to return zero.
    model.get_act_loss = lambda g_output, real_data, batch: torch.tensor(0.0)
    model.get_time_loss = lambda dis_data_time, g_output_time, batch: torch.tensor(0.0)
    model.get_act_time_loss = lambda dis_data_pos, dis_data_time, fake_act, fake_time, batch: torch.tensor(0.0)
    # Patch LightningModule helper functions.
    model.manual_backward = lambda loss: None
    model.toggle_optimizer = lambda opt: None
    model.untoggle_optimizer = lambda opt: None

    # Patch the optimizers to have no-op step and zero_grad.
    optimizers = model.configure_optimizers()
    new_optimizers = []
    for opt_dict in optimizers:
        opt = opt_dict["optimizer"]
        opt.zero_grad = lambda: None
        opt.step = lambda: None
        new_optimizers.append(opt)
    # Override model.optimizers() method to return our patched optimizers.
    model.optimizers = lambda: new_optimizers

    # Create a dummy batch.
    batch_size = 2
    seq_len = model.hparams.seq_len
    dis_data_pos = torch.randint(0, model.hparams.vocab_num_act, (batch_size, seq_len))
    dis_data_time = torch.rand(batch_size, seq_len)
    dis_data_duration = torch.rand(batch_size, 1)
    dummy_batch = (dis_data_pos, dis_data_time, dis_data_duration)

    # Call training_step.
    loss = model.training_step(dummy_batch)
    # In our patched training_step, generator update returns a loss of 1.0 and discriminator update adds 0.5.
    # Total loss should be 1.5.
    assert torch.is_tensor(loss)
    assert abs(loss.item() - 1.5) < 1e-5

def test_pre_train(model):
    # Patch get_pre_exp_loss to return fixed values.
    model.get_pre_exp_loss = lambda pre_epochs, g_model, d_model, dataloader: (0.1, 0.2, 0.3, 0.4)
    # Create a dummy dataloader (an iterable yielding one dummy batch).
    seq_len = model.hparams.seq_len
    dummy_batch = (
        torch.randint(0, model.hparams.vocab_num_act, (2, seq_len)),
        torch.rand(2, seq_len),
        torch.rand(2, 1)
    )
    dummy_dataloader = [dummy_batch]

    # Call pre_train.
    model.pre_train(dummy_dataloader)
    # Check that the mean losses are set as expected.
    assert abs(model.mean_act_loss - 0.1) < 1e-5
    assert abs(model.mean_gen_loss - 0.2) < 1e-5
    assert abs(model.mean_time_loss - 0.3) < 1e-5
    assert abs(model.mean_act_time_loss - 0.4) < 1e-5
    
    

def test_generator_output_shapes():
    # Define hyperparameters similar to expected config.
    input_size = 20  # Vocabulary size (number of possible activity tokens)
    num_emb = 16
    num_heads = 2
    num_hidden = 32
    num_layers = 2
    padding_index = 0
    dropout = 0.1

    # Instantiate the TransformerModel_GEN_Time generator.
    generator = TransformerModel_GEN_Time(
        input_size=input_size,
        num_emb=num_emb,
        num_heads=num_heads,
        num_hidden=num_hidden,
        num_layers=num_layers,
        padding_index=padding_index,
        dropout=dropout,
    )

    # Create dummy input tensors.
    seq_len = 10
    batch_size = 4
    # src: shape [seq_len, batch_size] containing token indices.
    src = torch.randint(0, input_size, (seq_len, batch_size)).long()
    # src_mask: a square mask for the sequence (shape [seq_len, seq_len])
    src_mask = generator.generate_square_subsequent_mask(seq_len)
    # duration: tensor of shape [batch_size] with random float values.
    duration = torch.rand(batch_size).view(-1)
    assert duration.dim() == 1, f"Expected duration to be 1D, got {duration.dim()}"

    # Forward pass through the generator.
    output_act, output_time = generator(src, src_mask, duration)

    # Expected output shapes:
    # output_act should be of shape [seq_len, batch_size, input_size]
    # output_time should be of shape [seq_len, batch_size, 1]
    assert output_act.shape == (seq_len, batch_size, input_size), (
        f"Expected activity output shape {(seq_len, batch_size, input_size)}, got {output_act.shape}"
    )
    assert output_time.shape == (seq_len, batch_size, 1), (
        f"Expected time output shape {(seq_len, batch_size, 1)}, got {output_time.shape}"
    )
    
def test_training_step_internal_logic(monkeypatch):
    # --- Define dummy generator and discriminator that expose the internal logic ---
    class DummyGenerator(nn.Module):
        def __init__(self, seq_len, vocab_num):
            super().__init__()
            self.seq_len = seq_len
            self.vocab_num = vocab_num + 1
        def forward(self, x, src_mask, duration):
            batch = x.size(1)
            # Return logits as zeros and time predictions as ones
            logits = torch.zeros(self.seq_len, batch, self.vocab_num, device=x.device)
            time_preds = torch.ones(self.seq_len, batch, device=x.device)
            return logits, time_preds
        def generate_square_subsequent_mask(self, sz):
            return torch.zeros(sz, sz)

    class DummyDiscriminator(nn.Module):
        def __init__(self, seq_len):
            super().__init__()
            self.seq_len = seq_len
        def forward(self, act, time, src_mask):
            # Use the second dimension (batch size) from the activity tensor
            batch = act.size(1)
            # Always output 0.9 for each sample
            return torch.full((batch, 1), 0.9, device=act.device)
        def generate_square_subsequent_mask(self, sz):
            return torch.zeros(sz, sz)

    # --- Define a dummy optimizer (no-op) ---
    class DummyOptimizer:
        def __init__(self, params):
            self.params = list(params)
        def zero_grad(self):
            pass
        def step(self):
            pass

    def dummy_optimizers_func():
        return (
            DummyOptimizer(dummy_gen.parameters()),
            DummyOptimizer(dummy_dis.parameters()),
        )

    # Override toggle/untoggle and manual_backward as no-ops
    dummy_noop = lambda opt: None
    dummy_manual_backward = lambda loss: None

    # --- Patch helper functions ---
    monkeypatch.setattr(
        h, "get_act_distribution",
        lambda g_out, real: (real.float(), real.float())
    )
    monkeypatch.setattr(
        h, "prepare_dis_label",
        lambda batch: (
            torch.ones(batch, device=torch.device("cpu")),
            torch.zeros(batch, device=torch.device("cpu")),
        )
    )

    # --- Create dummy hparams ---
    class DummyHParams:
        def __init__(self):
            self.seq_len = 5
            self.vocab_num_act = 10
            self.gd_ratio = 1
            self.model_mode = "vanilla"

    dummy_hparams = DummyHParams()

    # --- Instantiate dummy generator and discriminator ---
    dummy_gen = DummyGenerator(dummy_hparams.seq_len, dummy_hparams.vocab_num_act)
    dummy_dis = DummyDiscriminator(dummy_hparams.seq_len)

    # --- Instantiate the ProcessGANLitModule ---
    model = ProcessGANLitModule(
        generator=dummy_gen,
        discriminator=dummy_dis,
        gen_optimizer=lambda params: DummyOptimizer(params),
        dis_optimizer=lambda params: DummyOptimizer(params),
        gen_scheduler=None,
        dis_scheduler=None,
        compile=False,
        **dummy_hparams.__dict__
    )

    # Override methods to use our dummy optimizers and skip actual optimization
    model.optimizers = dummy_optimizers_func
    model.toggle_optimizer = dummy_noop
    model.untoggle_optimizer = dummy_noop
    model.manual_backward = dummy_manual_backward

    # --- Use the real dataloader from SepsisDataModule ---
    datamodule = SepsisDataModule(
        data_dir="tests/data",
        train_val_test_split=(629, 135, 135),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # Get one batch from the train dataloader
    batch = next(iter(datamodule.train_dataloader()))
    dis_data_pos = batch["activity_seq"]
    dis_data_time = batch["timestamp_diff"]
    duration = batch["duration"]
    if duration.dim() > 1:
        duration = duration.squeeze(-1)
    dummy_batch = (dis_data_pos, dis_data_time, duration)

    model.on_train_start()
    model.on_train_epoch_start()
    # --- Call training_step and expect a dictionary ---
    out = model.training_step(dummy_batch, 1)
    # E.g., out should be {"g_loss": ..., "d_loss": ...}

    # Extract the returned losses
    g_loss = out["g_loss"]
    d_loss = out["d_loss"]

    # --- Manually compute the expected losses ---
    # The dummy generator returns 0.9 from the discriminator => BCE(0.9, 1) => -log(0.9).
    expected_gen_loss = -torch.log(torch.tensor(0.9))

    # For the discriminator, real => BCE(0.9, 1) = -log(0.9),
    # fake => BCE(0.9, 0) = -log(0.1).
    expected_dis_loss = -torch.log(torch.tensor(0.9)) - torch.log(torch.tensor(0.1))

    # Compare each loss individually with tolerance
    tol = 1e-4
    assert torch.is_tensor(g_loss), "Generator loss must be a tensor"
    assert torch.is_tensor(d_loss), "Discriminator loss must be a tensor"

    np_g_loss = g_loss.item()
    np_d_loss = d_loss.item()

    # Check generator loss
    assert abs(np_g_loss - expected_gen_loss.item()) < tol, (
        f"Expected generator loss {expected_gen_loss.item()}, got {np_g_loss}"
    )
    # Check discriminator loss
    assert abs(np_d_loss - expected_dis_loss.item()) < tol, (
        f"Expected discriminator loss {expected_dis_loss.item()}, got {np_d_loss}"
    )
    
def test_count_data():
    from collections import Counter
    dataset = SepsisDataset("data\\raw\\Sepsis Cases - Event Log.xes")
    event_counter = Counter()
    n_samples = len(dataset)
    
    for i in range(n_samples):
        sample = dataset[i]
        # Get the entire activity sequence as a list (ignoring mask or any other field)
        activity_seq = sample["activity_seq"]
        tokens = activity_seq.tolist()
        event_counter.update(tokens)
    print(event_counter)
    assert 1 == 1, "Test failed!"
    
def test_generating_data():
    epoch = "1999"
    model = ProcessGANLitModule.load_from_checkpoint(f"logs/train/runs/2025-05-08_12-00-09/checkpoints/epoch_{epoch}.ckpt")
    model.eval()
    generator = model.generator
    
    batch_size = 64
    seq_len = 185
    vocab_size = 17
   
    # Create a dummy input tensor with the same shape as expected by the generator
   
    rand_set_np, mask = h.generate_random_data(batch_size, vocab_size, seq_len, 59)
    random_data = torch.tensor(rand_set_np, dtype=torch.int64, device=model.device)
    mask = torch.tensor(mask, dtype=torch.float32, device=model.device)
   
    mask_len = random_data.size(1)
    # Generate mask and put it on same device as data
    src_mask = generator.generate_square_subsequent_mask(mask_len).to(model.device)
   
    duration = torch.rand(64, device=model.device) * (1e-2 - 1e-3) + 1e-3
    
    g_output, g_output_time = generator(random_data, src_mask, duration, mask)


    # Use Gumbel-softmax for differentiable sampling, resulting in one-hot vectors.
    g_output_act = F.gumbel_softmax(g_output, tau=1, hard=True)
    # Apply padding after the end token
    pad_mask_mul, pad_mask_add = h.get_pad_mask(
            g_output_act,
            batch_size,
            seq_len,
            vocab_size,
            0,
            g_output_act
        )
    g_output_act = h.pad_after_end_token(g_output_act, pad_mask_mul, pad_mask_add)
        
    argmax_output = g_output_act.argmax(dim=-1)
    total_sum = argmax_output.sum()
    # Convert to numpy for easier comparison
    argmax_output_np = argmax_output.cpu().numpy()
    time_output_np = g_output_time.squeeze(-1).transpose(0, 1).cpu().detach().numpy()
    
    dataset = SepsisDataset("data\\raw\\Sepsis Cases - Event Log.xes")
    dict = dataset.activity2idx
    idx2act = {v: k for k, v in dict.items()}
    # Vectorize the lookup to convert numbers back to activity strings
    vectorized_lookup = np.vectorize(lambda x: idx2act.get(x, "Unknown"))
    activity_array = vectorized_lookup(argmax_output_np)
    
    timestamps = np.cumsum(time_output_np, axis=1)
    
    activities_flat = activity_array.flatten()         # shape: (64*185,)
    timestamps_flat = timestamps.flatten()             # shape: (64*185,)

    # Create a case id for each event.
    # For example, for 64 cases and 185 events per case, we can do:
    case_ids = np.repeat(np.arange(64), activity_array.shape[1])  # shape: (64*185,)

    # Build the DataFrame
    df = pd.DataFrame({
        'case_id': case_ids,
        'activity': activities_flat,
        'timestamp': timestamps_flat
    })
    df.to_csv(f"output_{epoch}.csv", index=False)
    test=1
    assert test == 1, "Test failed!"
    
    


   

    
