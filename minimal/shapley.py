import numpy as np
import shap
import seaborn as sn
import argparse
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm
import base
import make_results
import dataset
import exp

def make_shap(shap_exp):
    data,splits=shap_exp.get_data()
    clf_type=shap_exp.get_clf()
    def helper(split_i,clf_i):
        train,test=data.divide(split_i)
        if(shap_exp.k is None):
            background_data=train.X
        else:
            kmeans_summary = shap.kmeans( train.X, 
                                          shap_exp.k)
            background_data = kmeans_summary.data     
        explainer=shap.Explainer( clf_i.model.predict_proba,
                                  train.X)
        shap_values = explainer(test.X)#,max_evals=620)
        return shap_values.values
    print(shap_exp.out_path)
    base.make_dir(shap_exp.out_path)
    for i,split_i in enumerate(tqdm(splits)):
        out_i=f"{shap_exp.out_path}/{i}"
        clf_i,_=split_i.fit_clf(data,clf_type())
        values_i=helper(split_i,clf_i)
        np.savez(out_i, values_i)

def show_shapley(shapley_path):
    all_shap=[]
    for id_i, path_j in base.iter_files(shapley_path):
        shap_j=np.load(path_j)["arr_0"]
        all_shap.append(shap_j)
    shap_arr=np.concatenate(all_shap,axis=0)
    shap_matrix=np.mean(shap_arr,axis=0)
    print(shap_matrix.shape)
    show_heatmap( shap_matrix,
                  shapley_path)

def _show_shapley(shapley_path):
    for path_i in base.top_files(shapley_path):
        all_shap=[]
        for id_i, path_j in base.iter_files(path_i):
            shap_j=np.load(path_j)["arr_0"]
            all_shap.append(shap_j)
        shap_arr=np.concatenate(all_shap,axis=0)
        shap_matrix=np.mean(shap_arr,axis=0)
        print(shap_matrix.shape)
        show_heatmap( shap_matrix,
                      path_i)

def show_heatmap( matrix,
                  title,
                  out_path=None):
    sn.heatmap( matrix,
                cmap="YlGnBu",
                annot=False)#,
    plt.title(title)
    if(out_path):
        out_i=f"{out_path}/{title}"
        plt.tight_layout()
        plt.savefig(out_i,dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="zbiory_danych/cmc")
    parser.add_argument("--split_path", type=str, default="splits/cmc")
    parser.add_argument("--result_path", type=str, default="results/cmc")
    parser.add_argument("--clf", type=str, default="RF")
    parser.add_argument("--shapley_path", type=str,default="shapley/cmc")
    parser.add_argument("--k", type=str,default=100)
    parser.add_argument("--cmd", type=str,default="make")
    args=parser.parse_args()
    if(args.cmd=="make"):
        shap_exp=exp.ExpParams( args.data_path,
                                args.split_path,
                                args.shapley_path,
                                args.clf,
                                args.k
                              )
        make_shap(shap_exp)
    if(args.cmd=="show"):
        show_shapley(args.shapley_path)