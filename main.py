import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
tf.experimental.numpy.experimental_enable_numpy_behavior()
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("keirp/common_crawl_sample")

# Tokenize function for BERT
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer_bert(examples['text'], padding="max_length", truncation=True)

# Map dataset to tokenized format
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load GPT-2 tokenizer and model
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer_gpt2.eos_token_id)

# Define LNNM model (adjust the architecture based on your requirements)
class LNNM(tf.keras.Model):
    def __init__(self):
        super(LNNM, self).__init__()
        self.dense_layers = []
        for i in range(8):
            units = 80 - i * 10 if i * 10 < 80 else 10  # Decreasing number of units progressively
            self.dense_layers.append(tf.keras.layers.Dense(units, activation='relu'))

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # Using sigmoid activation for text output

    def call(self, inputs):
        x = inputs

        for layer in self.dense_layers:
            x = layer(x)

        output = self.output_layer(x)

        return output

# Instantiate the LNNM model
lnnm_model = LNNM()

# Define loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Training loop
num_epochs = 100000
batch_size = 32

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for i in range(0, len(tokenized_datasets['train']), batch_size):
        batch_inputs = tf.convert_to_tensor(tokenized_datasets['train']['input_ids'][i:i+batch_size])
        batch_attention_mask = tf.convert_to_tensor(tokenized_datasets['train']['attention_mask'][i:i+batch_size])

        with tf.GradientTape() as tape:
            # Generate output from GPT-2
            gpt_outputs = model_gpt2(input_ids=batch_inputs, attention_mask=batch_attention_mask, return_dict=True)
            gpt_logits = gpt_outputs.logits.shape[0]  # Get the batch

            # Example: Using BERT-like filter (replace with your actual filtering logic)
            filtered_labels = tf.random.uniform((batch_size, 1), minval=0, maxval=1)

            # Concatenate GPT-2 logits and filtered labels for LNNM input
            lnnm_inputs = tf.concat([gpt_logits, filtered_labels], axis=-1)

            # Make predictions with LNNM model
            lnnm_predictions = lnnm_model(lnnm_inputs)

            # Example: Compute loss (replace with your actual loss calculation)
            loss = loss_fn(filtered_labels, lnnm_predictions)

        gradients = tape.gradient(loss, lnnm_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, lnnm_model.trainable_variables))

        if (i // batch_size) % 10 == 0:
            print(f"Batch {i}/{len(tokenized_datasets['train'])}, Loss: {loss.numpy():.4f}")

print("Training completed.")
