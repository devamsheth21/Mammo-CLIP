import torch
import h5py
import os
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_utils import load_model
from datasets.data_utils import load_dataloader
sys.path.insert(0,'/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF')
sys.path.append('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/Mammo-CLIP/src/codebase/')
# from models.model_retrieval import ALBEF
# from models.tokenization_bert import BertTokenizer
from breastclip.model import build_model
from ruamel.yaml import YAML
from dataset import create_dataset, create_sampler, create_loader
import torch.nn.functional as F
import h5py

class Evaluation:
    def __init__(self, modelname, model=None, tokenizer=None, data_loader=None, config = None):
        self.modelname = modelname
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.config = config
        self.root_dir = '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/embeddings/'
        train_all = pd.read_json('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/uw_madison_train.json')
        vc = train_all.group_id.value_counts()
        self.rare_grp_ids = set(vc[20:].index)
        self.freq_ids = set(vc[:20].index)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def run_eval(self):
        k_values = [1,5,10]
        scores_i2t,scores_t2i,embeddings = self.evaluate()
        results_df = self.itm_eval_groups(scores_i2t, scores_t2i, self.data_loader.dataset.ann["group_id"].to_dict(), self.data_loader.dataset.ann["group_id"].to_dict(), k_values=k_values)
        csvpath =f"{self.root_dir}{self.modelname}-test-recall@1_5_10-ml256.csv"
        results_df.to_csv(csvpath)
        resultdf = self.create_metrics_table_with_ci(results_df,k_values=k_values)
        resultdf.to_csv(f'{self.modelname}-test-recall@1_5_10')
        return resultdf

    def get_embeddings_image(self, image):
        if self.modelname == 'ALBEF':
            image_feat = self.model.visual_encoder(image)
            image_embed = self.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)
        else:
            image_feat = self.model.encode_image(image)
            image_embed = F.normalize(self.model.image_projection(image_feat), dim=-1)
        return image_embed

    def get_embeddings_text(self, text_input):
        if self.modelname == 'ALBEF':
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(self.model.text_proj(text_feat[:, 0, :]))
        else:
            text_output = self.model.encode_text(text_input)
            text_embed = F.normalize(self.model.text_projection(text_output))
        return text_embed

    def evaluate(self):
        device = self.device
        csvpath =f"{self.root_dir}{self.modelname}-test-recall@1_5_10-ml256.csv"
        if os.path.exists(csvpath):
            df = pd.read_csv(csvpath)
            evaluator.create_metrics_table_with_ci(df,k_values) 
            
        embeddings = {}

        print('Computing features for evaluation...')
        start_time = time.time()

    
        self.root_dir = '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/embeddings'
        savepath = f'{self.root_dir}/{self.modelname}img_feats_g20_mammo_ML256.npz'
        embedpath = {'albef': '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/img_feats_g20_b3_new_preprocess_unique_ML256.hdf5',
                     'mammoclip': '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/embeddings/img_feats_g20_mammo_ML256.hdf5',
                     'medimageinsight': '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/embeddings/img_feats_g20_MedImageInsight.h5',
                     'multiview': '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/Mammo-CLIP/src/codebase/finetune/embeddings/embeddings_exp1.h5'
                     }
        hpath = embedpath.get(self.modelname.lower(), 'None')
        if os.path.exists(hpath):
            print(f"Loading saved embeddings from: {hpath}")
            with h5py.File(hpath, 'r') as file:
                availablekeys = list(file.keys())
                textkey = [key for key in availablekeys if key.startswith('t')][0]
                availablekeys.remove(textkey)
                imgkey = [key for key in availablekeys if ('feat' in key or 'embedding' in key)][0]
                print("image key", imgkey, "text key", textkey)
                textfeat = np.array(file[textkey])
                imgfeat = np.array(file[imgkey])
                text_embeds = torch.Tensor(textfeat[:imgfeat.shape[0]])
                image_embeds = torch.Tensor(imgfeat)
        else:
            if self.modelname == "ALBEF":
                self.model, self.tokenizer = self.load_albef()
            elif self.modelname.lower() == 'mammoclip':
                self.model, self.tokenizer = self.load_mammoclip_model()
            elif self.modelname.lower() == 'multiview':
                if not self.model :
                    self.model, self.tokenizer = self.load_mutliview()
            else:
                self.model, self.tokenizer = None, None
            self.model.eval()
            self.model.to(device)

            texts = self.data_loader.dataset.text
            num_text = len(texts)
            print(num_text, self.data_loader.dataset.__len__())
            text_bs = 256
            text_embeds = []
            for i in tqdm(range(0, num_text, text_bs), desc="Processing Texts"):
                text = texts[i: min(num_text, i + text_bs)]
                text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=256,
                                            return_tensors="pt").to(device)
                text_embed = self.get_embeddings_text(text_input)
                text_embeds.append(text_embed)
            text_embeds = torch.cat(text_embeds, dim=0)

            image_embeds = []
            for image, img_id in tqdm(self.data_loader, desc="Processing Images"):
                image = image.to(device)
                image_embed = self.get_embeddings_image(image)
                image_embeds.append(image_embed)
            image_embeds = torch.cat(image_embeds, dim=0)

            print(f"Text embed shape: {text_embeds.shape}, image embed shape: {image_embeds.shape}")

            embeddings["text_embeddings"] = text_embeds.cpu().numpy()
            embeddings["image_embeddings"] = image_embeds.cpu().numpy()
            with h5py.File(hpath, "w") as f:
                f.create_dataset("txt_features", data=embeddings["text_embeddings"])
                f.create_dataset("features", data=embeddings["image_embeddings"])

        available_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"Available GPU memory: {available_memory / (1024 * 1024 * 1024):.2f} GB")

        embedding_size = image_embeds.shape[0] * image_embeds.shape[1] * 4
        required_memory = embedding_size * 2
        print(f"Required GPU memory: {required_memory / (1024 * 1024 * 1024):.2f} GB")

        if available_memory < required_memory:
            print("Insufficient GPU memory. Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_cached()
            print("GPU memory cleared.")
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
        else:
            print("Sufficient GPU memory available.")
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
            print("----------Shape-----------: ", image_embeds.shape, text_embeds.shape)
            batch_size = 256
            num_images = image_embeds.shape[0]
            num_texts = text_embeds.shape[0]
            sims_matrix = torch.zeros((num_images, num_texts))
            for i in range(0, num_images, batch_size):
                image_batch = image_embeds[i:i+batch_size]
                sims_batch = torch.matmul(image_batch, text_embeds.t())
                sims_matrix[i:i+batch_size] = sims_batch.cpu()

        score_matrix_i2t = torch.full((len(self.data_loader.dataset), len(text_embeds)), -100.0)
        score_matrix_t2i = torch.full((len(text_embeds), len(self.data_loader.dataset)), -100.0)

        topk_sim, topk_idx = torch.topk(sims_matrix, k=20, dim=1)
        score_matrix_i2t.scatter_(1, topk_idx, topk_sim)

        sims_matrix_t2i = sims_matrix.t()

        topk_sim_t2i, topk_idx_t2i = torch.topk(sims_matrix_t2i, k=20, dim=1)
        score_matrix_t2i.scatter_(1, topk_idx_t2i, topk_sim_t2i)

        score_matrix_i2t = score_matrix_i2t.cpu().numpy()
        score_matrix_t2i = score_matrix_t2i.cpu().numpy()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time: {}'.format(total_time_str))

        return topk_idx, topk_idx_t2i, embeddings

    def load_albef(self, model_name_or_path="./output/Retrieval_g20_b3_new_preprocess/checkpoint_best.pth"):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = ALBEF(config=self.config, text_encoder='bert-base-uncased', tokenizer=tokenizer)
        checkpoint = torch.load(model_name_or_path, map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        return model, tokenizer

    def load_mammoclip_model(self, checkpoint_path='/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/checkpoints/b5-model-best-epoch-7.tar',
                             cache_dir="/home/ixb004/.cache/huggingface/hub/"):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if cache_dir:
            ckpt["config"]["tokenizer"]['cache_dir'] = cache_dir
            ckpt["config"]["model"]["text_encoder"]["cache_dir"] = cache_dir
            ckpt["config"]["model"]["text_encoder"]["name"] = os.path.join(cache_dir,
                                                                            "models--emilyalsentzer--Bio_ClinicalBERT/")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='emilyalsentzer/Bio_ClinicalBERT'
        )
        model = build_model(
            model_config=ckpt["config"]["model"],
            loss_config=ckpt["config"]["loss"],
            tokenizer=tokenizer
        )
        model.load_state_dict(ckpt["model"], strict=False)

        return model, tokenizer
    def load_multiview(self, checkpoint_path):
        model, tokenizer = load_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        
        return model, tokenizer
    
    def calculate_metrics(self, df, score_col, group_type=None):
        mask = pd.notna(df[score_col])
        if group_type:
            mask = mask & (df['group_type'] == group_type)

        scores = df[mask][score_col]
        if len(scores) == 0:
            return {'recall': 0, 'ci_lower': 0, 'ci_upper': 0, 'count': 0}

        mean_recall = scores.mean()

        ci = stats.t.interval(confidence=0.95,
                              df=len(scores) - 1,
                              loc=mean_recall,
                              scale=stats.sem(scores))

        return {
            'recall': mean_recall,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'count': len(scores)
        }

    def itm_eval_groups(self, scores_i2t, scores_t2i, txt2grp, img2grp, k_values=[1, 5, 10]):
        rareids = {"img": [], "txt": []}
        img_ids = list(img2grp.keys())
        txt_ids = list(txt2grp.keys())
        img_group_ids = [img2grp[img_id] for img_id in img_ids]
        num_samples = len(scores_i2t)
        results = {
            'index': list(range(num_samples)),
            'img_id': img_ids,
            'txt_id': txt_ids,
            'i2t_recall': np.zeros(num_samples),
            't2i_recall': np.zeros(num_samples),
            'group_id': img_group_ids,
            'group_type': ['rare' if gid in self.rare_grp_ids else 'common' for gid in img_group_ids] if self.rare_grp_ids else [
                'common'] * num_samples
        }

        for k in k_values:
            results[f'i2t_recall@{k}'] = np.zeros(num_samples)
            results[f't2i_recall@{k}'] = np.zeros(num_samples)

        for index, score in enumerate(tqdm(scores_i2t, total=len(scores_i2t), desc="Processing i2t")):
            for k in k_values:
                ranked_groups = set(txt2grp[int(txt_id)] for txt_id in score[:k] if int(txt_id) in txt2grp)
                imgid = img_ids[index]
                gt_groups = set([img2grp[imgid]])
                recall = len(gt_groups.intersection(ranked_groups)) / float(len(gt_groups))
                results[f'i2t_recall@{k}'][index] = recall

        for index, score in enumerate(tqdm(scores_t2i, total=len(scores_t2i), desc="Processing t2i")):
            for k in k_values:
                ranked_groups = set(img2grp[int(img_id)] for img_id in score[:k] if int(img_id) in img2grp)
                txtid = txt_ids[index]
                gt_groups = set([txt2grp[txtid]])
                recall = len(gt_groups.intersection(ranked_groups)) / float(len(gt_groups))
                results[f't2i_recall@{k}'][index] = recall

        results_df = pd.DataFrame(results)
        return results_df

    def load_dataset(self, config):
        TOP_N = 500
        val, train, test_dataset = create_dataset('re', config)
        samplers = [None] * 3
        data_loader = create_loader([test_dataset, train, val], samplers,
                                    batch_size=[2] * 3,
                                    num_workers=[4] * 3,
                                    is_trains=[False] * 3,
                                    collate_fns=[None] * 3)
        return data_loader[0]

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def save_json(self, data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f)

    def create_metrics_table_with_ci(self, df, k_values):
        # Initialize metrics data without counts initially
        metrics_data = {
            'Metric': [],
            f'{self.modelname} Rare': [],
            f'{self.modelname} Freq': [],
            f'{self.modelname} All': []
        }
        
        # Define metric order
        metric_order = [
            ('i2t', 1),
            ('i2t', 5),
            ('i2t', 10),
            ('t2i', 1),
            ('t2i', 5),
            ('t2i', 10)
        ]
        
        # Calculate metrics
        for direction, k in metric_order:
            metric_name = f"{direction.upper()} Recall@{k}"
            metrics_data['Metric'].append(metric_name)
            
            # Calculate for rare groups
            rare_metrics = self.calculate_metrics(df, f'{direction}_recall@{k}', 'rare')
            metrics_data[f'{self.modelname} Rare'].append(
                f"{rare_metrics['recall']:.4f} [{rare_metrics['ci_lower']:.4f}, {rare_metrics['ci_upper']:.4f}]"
            )
            
            # Calculate for frequent groups
            freq_metrics = self.calculate_metrics(df, f'{direction}_recall@{k}', 'common')
            metrics_data[f'{self.modelname} Freq'].append(
                f"{freq_metrics['recall']:.4f} [{freq_metrics['ci_lower']:.4f}, {freq_metrics['ci_upper']:.4f}]"
            )
            
            # Calculate for all data
            all_metrics = self.calculate_metrics(df, f'{direction}_recall@{k}')
            metrics_data[f'{self.modelname} All'].append(
                f"{all_metrics['recall']:.4f} [{all_metrics['ci_lower']:.4f}, {all_metrics['ci_upper']:.4f}]"
            )
        
        # Create DataFrame
        results_df = pd.DataFrame(metrics_data)
        
        # Get counts from the first metric calculation
        rare_count = rare_metrics['count']
        freq_count = freq_metrics['count']
        all_count = all_metrics['count']
        
        # Rename columns to include counts
        results_df = results_df.rename(columns={
            f'{self.modelname} Rare': f'{self.modelname} Rare (n={rare_count})',
            f'{self.modelname} Freq': f'{self.modelname} Freq (n={freq_count})',
            f'{self.modelname} All': f'{self.modelname} All (n={all_count})'
        })
        
        return results_df
    def report_metrics(self, df, k_values=[1, 5, 10]):
        results = []
        for k in k_values:
            metrics = {
            'i2t': {
                'all': self.calculate_metrics(df, f'i2t_recall@{k}'),
                'rare': self.calculate_metrics(df, f'i2t_recall@{k}', 'rare'),
                'common': self.calculate_metrics(df, f'i2t_recall@{k}', 'common')
            },
            't2i': {
                'all': self.calculate_metrics(df, f't2i_recall@{k}'),
                'rare': self.calculate_metrics(df, f't2i_recall@{k}', 'rare'),
                'common': self.calculate_metrics(df, f't2i_recall@{k}', 'common')
            }
            }
            for group_type in ['rare', 'common']:
                m = metrics['i2t'][group_type]
                results.append({
                    'metric': f"i2t_recall@{k}",
                    f"{modelname}-{group_type}": m['recall']
                })
                m = metrics['t2i'][group_type]
                results.append({
                    'metric': f"t2i_recall@{k}",
                    f"{modelname}-{group_type}": m['recall']
                })

        results_df = pd.DataFrame(results)
        return results_df

def main(config,modelname):
    evaluator = Evaluation(modelname=modelname,config=config)
    k_values = [1,5,10]
    csvpath =f"{self.root_dir}{modelname}-test-recall@1_5_10-ml256.csv"
    if os.path.exists(csvpath):
        df = pd.read_csv(csvpath)
        evaluator.reportmetrics(df,k_values)
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dataloader = load_dataset(config)
    
    scores_i2t,scores_t2i,embeddings = evaluator.evaluate()
    dataloader = load_dataloader(config, tokenizer=evaluator.tokenizer, split=None)
    results_df = evaluator.itm_eval_groups(scores_i2t, scores_t2i, dataloader.dataset.txt2grp, dataloader.dataset.img2grp, k_values=k_values)
    results_df.to_csv(csvpath)
    evaluator.reportmetrics(results_df,k_values=k_values)



    # if modelname =="ALBEF":
    #     model,tokenizer = load_albef()
    # elif modelname.lower()== 'mammoclip':
    #     model,tokenizer = load_mammoclip_model()
    # else:
    #     model,tokenizer = None,None
    
    # scores_i2t,scores_t2i,embeddings = evaluation( dataloader, tokenizer, device, config, modelname)
    # results_df = itm_eval_groups(scores_i2t, scores_t2i, dataloader.dataset.txt2grp, dataloader.dataset.img2grp, k_values=k_values, rare_ids = self.rare_grp_ids)
    # results_df.to_csv(f"{self.root_dir}{modelname}-test-recall@1_5_10-ml256.csv")
    # reportmetrics(results_df,k_values=k_values)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default = 'ALBEF')
    parser.add_argument('--config_path', default = './finetune-config.yaml') 
    #'/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/configs/Retrieval_coco.yaml'
    args = parser.parse_args()

    root_dir = '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/embeddings/'
    yaml = YAML(typ='safe') 
    with open(args.config_path, 'r') as file:
        config = yaml.load(file)
    print(config)
    main(config,args.modelname)