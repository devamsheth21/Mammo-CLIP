import pandas as pd
from tqdm import tqdm
import os
# Read findings with birads
df = pd.read_csv('/mnt/PURENFS/SalkowskiPreprocessedBreast/data/findings_new_birads_jg_dbt.csv', encoding_errors="ignore")

# Clean data
df = df.rename(columns={"Accession Number": "AccessionNumber", "Findings": "findings", "BI-RADS:":"birads"})
df = df.fillna(' ')
df = df[df['findings'] != ' ']

# Remove samples with noisy birads
df = df[~df['birads'].isin(['n', '#VALUE!', 'P', 'A', '.'])]
df['birads'] = pd.to_numeric(df['birads'])
df = df[~df['findings'].isin(["#VALUE!"])]

df_img = pd.read_csv('/mnt/PURENFS/SalkowskiPreprocessedBreast/data/final_df.csv')

df_txt = df[df['AccessionNumber'].isin(df_img['AccessionNumber'].unique())]

df_img_2 = df_img[df_img.AccessionNumber.isin(df_txt.AccessionNumber.unique())]

path2 = '/mnt/PURENFS/SalkowskiPreprocessedBreast/preproceessed/'

fpaths = []
for index, row in tqdm(df_img_2.iterrows()):
    img_path = os.path.join(path2, '_'.join(row.filepath.split('/')[-2:]))
    fpaths.append(img_path)
df_img_2['processed_filepath'] = fpaths   

# Pivot the table to get all info in one row
pivoted_df = df_img_2[['AccessionNumber','ImageLaterality','processed_filepath','filepath']].pivot_table(index='AccessionNumber', 
                               columns=['ImageLaterality'], 
                               values=['processed_filepath','filepath'], 
                               aggfunc='first')
pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]
pivoted_df.reset_index(inplace=True)

# Add Missing flag
pivoted_df['Missing'] = pivoted_df.apply(lambda row: 'L' if pd.isnull(row.get('processed_filepath_L')) 
                                         else 'R' if pd.isnull(row.get('processed_filepath_R')) 
                                         else 'None', axis=1)
pivoted_df['Laterality'] = pivoted_df.apply(lambda row: 'Both' if all(pd.notnull([row.get('processed_filepath_L'), row.get('processed_filepath_R')])) 
                                            else 'L' if pd.notnull(row.get('processed_filepath_L')) 
                                            else 'R', axis=1)

merged_df = pivoted_df.merge(df_txt, on="AccessionNumber")

groupsdf = pd.read_json('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/reports_with_correct_groups.json')[['GROUP','AccessionNumber','DENSITY_CATEGORY']]
merged_df = merged_df.merge(groupsdf, on='AccessionNumber')

groupsdf = pd.read_json('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/uw_madison_test_with_groupids.json')
groupsdf['AccessionNumber'] = groupsdf.image_id.apply(lambda x : x.split('_')[0])
groupsdf = groupsdf[['AccessionNumber','group_id']]
merged_df = merged_df.merge(groupsdf, on='AccessionNumber')

merged_df.rename(columns={'GROUP' : 'group'}, inplace=True)

# Add density label

density2label = {
    'scattered fibroglandular densities': 0,
    'heterogeneously dense': 1,
    'fatty': 2,
    'extremely dense': 3
}

merged_df['density'] = merged_df['DENSITY_CATEGORY'].map(density2label).fillna(-1)

# Split data

merged_df['split'] = 'train'
dfval = pd.read_json('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/uw_madison_val.json')
dfval['acc'] = dfval.image_id.apply(lambda x : x.split('_')[0])
merged_df.loc[merged_df['AccessionNumber'].isin(dfval['acc']), 'split'] = "val"
df2 = pd.read_json('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/uw_madison_test_1000.json')
df2['acc'] = df2.image_id.apply(lambda x : x.split('_')[0])
merged_df.loc[merged_df.AccessionNumber.isin(df2['acc']), 'split'] = "test"
merged_df.to_csv('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/data/uw_madison_pretrain_individual_lateralityv2.csv')
