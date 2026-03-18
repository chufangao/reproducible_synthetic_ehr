"""
Docstring for github.ehr_syn_repro.flattened_baselines

This script processes MIMIC EHR data to create flattened representations
and applies the GReaT generative model to synthesize new data samples.
Other baseline models can be added similarly.

Usage:
CUDA_VISIBLE_DEVICES=0 python flattened_baselines.py --MIMIC_DATA_PATH <path_to_mimic_data> \
    --train_patients_path <path_to_train_ids> \
    --test_patients_path <path_to_test_ids> \
    --save_synthetic_path <path_to_save_synthetic_data> \
    --mode great
"""

import pandas as pd
import os
import argparse

import os
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm, trange

# ==========================================
# 1. Prepare Data
# ==========================================

# # Raw data: List of patient histories
# raw_data = [
#     ["code1", "code2", "VISIT_DELIM", "code2", "code4", "code5", "VISIT_DELIM", "code1", "code1"],
#     ["code3", "code1", "VISIT_DELIM", "code5", "code9", "VISIT_DELIM", "code2"],
#     ["code1", "code2", "VISIT_DELIM", "code2", "code4"]
# ]

# # Flatten data into strings for the tokenizer trainer (space-separated)
# # We treat each code as a "word"
# text_data = [" ".join(seq) for seq in raw_data]


# Configuration
VISIT_DELIM = "VISIT_DELIM"

def tabular_to_sequences(df):
    """
    Converts a long-form EHR DataFrame into patient sequences.
    Order: Group by SUBJECT_ID -> Sort/Group by HADM_ID -> Collect Codes
    """
    # 1. Clean data: Ensure IDs are integers/strings (handle the .0 issue)
    df = df.copy()
    df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(int)
    df['HADM_ID'] = df['HADM_ID'].astype(int)
    df['ICD9_CODE'] = df['ICD9_CODE'].astype(str)

    # 2. Aggregation Helper
    # This creates a list of codes for each admission
    # Note: In real scenarios, ensure you sort by ADMITTIME before this step!
    visits = df.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].apply(list).reset_index()

    # 3. Create Patient Sequences
    # Group visits by patient and join them with the delimiter
    patient_seqs = []
    
    # We iterate by patient to preserve structure
    for subject_id, subject_data in visits.groupby('SUBJECT_ID'):
        # subject_data is a DataFrame of visits for one patient
        patient_history = []
        
        for codes_list in subject_data['ICD9_CODE']:
            # Join codes within one visit (e.g., "code1 code2")
            visit_str = " ".join(codes_list)
            patient_history.append(visit_str)
        
        # Join all visits with the delimiter
        full_seq = f" {VISIT_DELIM} ".join(patient_history)
        patient_seqs.append(full_seq)

    return patient_seqs

def sequences_to_tabular(sequences):
    """
    Converts list of text sequences back into a long-form DataFrame.
    Generates synthetic SUBJECT_ID and HADM_ID.
    """
    data_rows = []

    for subj_idx, seq in enumerate(sequences):
        # 1. Split sequence into visits
        # We strip to remove leading/trailing spaces
        visits = seq.strip().split(VISIT_DELIM)
        
        for hadm_idx, visit_str in enumerate(visits):
            # 2. Split visit into individual codes
            codes = visit_str.strip().split()
            
            # 3. Create a row for each code
            for code in codes:
                if code: # Check if code is not empty string
                    data_rows.append({
                        'SUBJECT_ID': subj_idx,       # Synthetic Patient ID
                        'HADM_ID': hadm_idx, # Synthetic Visit ID
                        'ICD9_CODE': code
                    })

    return pd.DataFrame(data_rows)

# ==========================================
# 3. Create PyTorch Dataset
# ==========================================
class EHRDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.input_ids = []
        
        for txt in txt_list:
            # Tokenize and truncate/pad
            encodings = tokenizer(
                txt, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings["input_ids"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument("--MIMIC_DATA_PATH", type=str, default="/home/chufan2/github/ehr_syn_repro/", help="Path to the MIMIC data directory.")
    parser.add_argument("--train_patients_path", type=str, default="/home/chufan2/github/ehr_syn_repro/data/train_patient_ids.txt", help="Path to the train patient IDs file.")
    parser.add_argument("--test_patients_path", type=str, default="/home/chufan2/github/ehr_syn_repro/data/test_patient_ids.txt", help="Path to the test patient IDs file.")
    parser.add_argument("--save_synthetic_path", type=str, default="/home/chufan2/github/ehr_syn_repro/synthetic_data/", help="Path to save synthetic data.")
    parser.add_argument("--mode", type=str, default="great", help="Which baseline model to use.")
    args = parser.parse_args()

    # assert args.mode in ['great', 'realtabformer']

    admissions_df = pd.read_csv(os.path.join(args.MIMIC_DATA_PATH, 'ADMISSIONS.csv'))
    patients_df = pd.read_csv(os.path.join(args.MIMIC_DATA_PATH, 'PATIENTS.csv'))
    diagnoses_df = pd.read_csv(os.path.join(args.MIMIC_DATA_PATH, 'DIAGNOSES_ICD.csv'))

    print(f"Admissions shape: {admissions_df.shape}")
    print(f"Patients shape: {patients_df.shape}")
    print(f"Diagnoses shape: {diagnoses_df.shape}")

    admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
    patients_df['DOB'] = pd.to_datetime(patients_df['DOB'])
    # Calculate age at first admission
    first_admissions = admissions_df.loc[
        admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
    ][['SUBJECT_ID', 'ADMITTIME']]

    demo_df = pd.merge(
        patients_df[['SUBJECT_ID', 'GENDER', 'DOB']],
        first_admissions,
        on='SUBJECT_ID',
        how='inner'
    )

    demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)
    demo_df['AGE'] = demo_df['AGE'].apply(lambda x: 90 if x > 89 else x)

    # Merge admissions with diagnoses
    admissions_info = admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    merged_df = pd.merge(
        admissions_info,
        diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE',]],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='inner'
    )

    # Merge with demographics
    final_df = pd.merge(
        merged_df,
        demo_df[['SUBJECT_ID', 'AGE', 'GENDER']],
        on='SUBJECT_ID',
        how='left'
    )

    # Sort chronologically
    final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME',], inplace=True)

    # map all HADM_ID to sequential visit IDs per SUBJECT_ID, starting from 0
    final_df["VISIT_ID"] = final_df.groupby("SUBJECT_ID")["HADM_ID"].transform(lambda x: pd.factorize(x)[0])

    final_df['SUBJECT_ID'] = final_df['SUBJECT_ID'].astype(str)
    final_df['HADM_ID'] = final_df['HADM_ID'].astype(float)
    final_df['ICD9_CODE'] = final_df['ICD9_CODE'].astype(str)

    final_df = final_df.drop(columns=['ADMITTIME', 'AGE', 'GENDER', 'VISIT_ID'])
    final_df = final_df.dropna()

    # split by train and test
    train_patient_ids = pd.read_csv(args.train_patients_path, header=None)[0].astype(str)
    test_patients_ids = pd.read_csv(args.test_patients_path, header=None)[0].astype(str)
    train_ehr = final_df[final_df['SUBJECT_ID'].isin(train_patient_ids)].reset_index(drop=True)
    test_ehr = final_df[final_df['SUBJECT_ID'].isin(test_patients_ids)].reset_index(drop=True)

    
    final_df = final_df.dropna()
    final_df['ICD9_CODE'] = final_df['ICD9_CODE'].astype(str)
    result_df = pd.crosstab(final_df['SUBJECT_ID'], final_df['ICD9_CODE']).reset_index()
    result_df.columns.name = None
    result_df = result_df.drop(columns=['nan'])
    result_df = result_df.astype(int)

    # select train and test splits
    result_df['SUBJECT_ID'] = result_df['SUBJECT_ID'].astype(int).astype(str)
    train_flattened = result_df[result_df['SUBJECT_ID'].isin(train_patient_ids)].reset_index(drop=True)
    test_flattened = result_df[result_df['SUBJECT_ID'].isin(test_patients_ids)].reset_index(drop=True)

    # drop SUBJECT_ID column for model input
    train_flattened = train_flattened.drop(columns=['SUBJECT_ID'])
    test_flattened = test_flattened.drop(columns=['SUBJECT_ID'])

    # # drop columns with all zero values
    # train_flattened = train_flattened.loc[:, (train_flattened != 0).any(axis=0)]
    # test_flattened = test_flattened.loc[:, (test_flattened != 0).any(axis=0)]
    # # write to csv
    # train_flattened.to_csv(MIMIC_DATA_PATH + 'train_flattened_ehr.csv', index=False)
    # test_flattened.to_csv(MIMIC_DATA_PATH + 'test_flattened_ehr.csv', index=False)

    if args.mode=='great': # super slow because it has to query for every row
        import be_great
        # https://github.com/tabularis-ai/be_great
        # from sklearn.datasets import fetch_california_housing
        # data = fetch_california_housing(as_frame=True).frame
        model = be_great.GReaT(llm='tabularisai/Qwen3-0.3B-distil', batch_size=512, epochs=2, dataloader_num_workers=4, fp16=True)
        model.fit(train_flattened)
        # model.save("my_directory")  # saves a "model.pt" and a "config.json" file
        model.save(os.path.join(args.save_synthetic_path, "great"))
        synthetic_data = model.sample(n_samples=10000)
        synthetic_data.to_csv(os.path.join(args.save_synthetic_path, 'great/great_synthetic_flattened_ehr.csv'), index=False)
    elif args.mode=='realtabformer': # broken
        import realtabformer
        # Non-relational or parent table.
        rtf_model = realtabformer.REaLTabFormer(
            model_type="tabular",
            gradient_accumulation_steps=4,
            logging_steps=100)
        # Fit the model on the dataset.
        # Additional parameters can be
        # passed to the `.fit` method.
        rtf_model.fit(train_flattened)
        # Save the model to the current directory.
        # A new directory `rtf_model/` will be created.
        # In it, a directory with the model's
        # experiment id `idXXXX` will also be created
        # where the artefacts of the model will be stored.
        rtf_model.save(os.path.join(args.save_synthetic_path, "realtabformer/rtf_model"))
        # Load the saved model. The directory to the
        # experiment must be provided.
        rtf_model2 = realtabformer.REaLTabFormer.load_from_dir(
            # path=os.path.join(args.save_synthetic_path, "realtabformer/rtf_model")
            path="/home/chufan2/github/ehr_syn_repro/rtf_checkpoints/checkpoint-7115"
            )
        # Generate synthetic data with the same
        # number of observations as the real dataset.
        samples = rtf_model.sample(n_samples=10000)
        samples.to_csv(os.path.join(args.save_synthetic_path, 'realtabformer/realtabformer_synthetic_flattened_ehr.csv'), index=False)
    elif args.mode in ['ctgan','tvae']:
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
        # 1. auto-detect metadata based in your data
        # train_flattened_subset = train_flattened.sample(n=10000, random_state=42)
        metadata = Metadata.detect_from_dataframe(data=train_flattened)
        for column in train_flattened.columns:
            metadata.update_column(
                column_name=column,
                sdtype='numerical',
            )
        # # 2. carefully inspect and update your metadata
        # metadata.visualize()

        if args.mode=='ctgan':
            synthesizer = CTGANSynthesizer(metadata, epochs=2, batch_size=64)
        elif args.mode=='tvae':
            synthesizer = TVAESynthesizer(metadata, epochs=2, batch_size=64)
        synthesizer.fit(train_flattened)
        synthesizer.save(filepath=os.path.join(args.save_synthetic_path, f'{args.mode}/synthesizer.pkl'))
        synthetic_data = synthesizer.sample(num_rows=10000)
        synthetic_data.to_csv(os.path.join(args.save_synthetic_path, f'{args.mode}/{args.mode}_synthetic_flattened_ehr.csv'), index=False)
    elif args.mode in ['transformer_baseline']:
        # ==========================================
        # 2. Build Custom Tokenizer
        # ==========================================
        max_gens = 10000
        gen_batch_size = 512
        train_batch_size = 64
        num_train_epochs = 50
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        text_data = tabular_to_sequences(train_ehr[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']])
        max_len = max([len(seq.split()) for seq in text_data])
        print(f"Max sequence length in training data: {max_len}")
        # We use a WordLevel model so "code1" is not split into "code" + "1"
        tokenizer_obj = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

        # Special tokens for training
        special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)

        # Train the tokenizer on our data
        tokenizer_obj.train_from_iterator(text_data, trainer=trainer)

        # Add post-processing to automatically add BOS/EOS tokens
        tokenizer_obj.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer_obj.token_to_id("[BOS]")),
                ("[EOS]", tokenizer_obj.token_to_id("[EOS]")),
            ],
        )

        # Wrap in Hugging Face's PreTrainedTokenizerFast wrapper
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        dataset = EHRDataset(text_data, tokenizer)

        print(f"Vocab Size: {len(tokenizer)}")

        # ==========================================
        # 4. Initialize Decoder Model (GPT-2 style)
        # ==========================================
        # We use a small configuration for this baseline
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=128,  # Max sequence length
            n_ctx=512,
            n_embd=512,
            n_layer=8,
            n_head=8,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        model = GPT2LMHeadModel(config).to(device)

        # ==========================================
        # 5. Training
        # ==========================================
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False # False ensures we do Causal Language Modeling (Decoder)
        )

        training_args = TrainingArguments(
            output_dir="./synthetic_ehr_model",
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,             
            per_device_train_batch_size=train_batch_size,
            logging_steps=100,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            use_cpu=not torch.cuda.is_available() # Remove if you have a GPU
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()

        # save model
        trainer.save_model(os.path.join(args.save_synthetic_path, "transformer_baseline/transformer_baseline_model_final"))
        # ==========================================
        # 6. Generate Synthetic EHRs
        # ==========================================
        print("\n--- Generating Synthetic EHRs ---")

        model.eval()

        # Start generation with the [BOS] token
        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)

        all_syn_dfs = []
        start_patient_id = 0
        for start_idx in trange(0, max_gens, gen_batch_size):
            end_idx = min(start_idx + gen_batch_size, max_gens)
            batch_size = end_idx - start_idx

            # Prepare batch input_ids
            batch_input_ids = torch.tensor([[tokenizer.bos_token_id]] * batch_size).to(device)
            # Generate sequences
            generated_ids = model.generate(
                batch_input_ids,
                max_length=max_len,
                # num_return_sequences=1,
                do_sample=True,      # Add randomness so we don't get identical outputs
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode back to codes
            all_decoded = []
            for i, sample in enumerate(generated_ids):
                decoded = tokenizer.decode(sample, skip_special_tokens=True)
                all_decoded.append(decoded)
                # print(f"Synthetic Patient {i+1}: {decoded}")

            syn_df = sequences_to_tabular(all_decoded)
            syn_df['SUBJECT_ID'] += start_patient_id
            start_patient_id += batch_size
            all_syn_dfs.append(syn_df)

        all_syn_df = pd.concat(all_syn_dfs, ignore_index=True)
        print("\nSynthetic DataFrame:")
        print(all_syn_df)
        all_syn_df.to_csv(os.path.join(args.save_synthetic_path, 'transformer_baseline/transformer_baseline_synthetic_ehr.csv'), index=False)
    else:
        raise NotImplementedError