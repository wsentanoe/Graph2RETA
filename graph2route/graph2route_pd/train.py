# -*- coding: utf-8 -*-
from evaluation.eval_route import *
import torch.nn.functional as F
# os.environ['MKL_SERVICE_FORCE_INTEL']='1'
# os.environ['MKL_THREADING_LAYER']='GNU'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,4,5,6,7'
from tqdm import tqdm
from evaluation.eval_route import Metric
from my_utils.utils import run, dict_merge
from my_utils.utils import get_nonzeros
from graph2route.graph2route_pd.model import Graph2RouteDataset
import pickle
from pathlib import Path


def collate_fn(batch):
    return batch


def process_batch(batch, model, device, pad_vaule):
    E_ed, V, V_reach_mask, label_len, label, V_pt, V_ft, start_idx, \
        E_sd, V_dt, V_num, E_mask, V_dispatch_mask, pt_dif, dt_dif, cou, A, unpick_x, unpick_len, last_x, last_len, \
        eta_npa, order_np, index_np, index = zip(*batch)

    V = np.array(V)
    V_reach_mask = np.array(V_reach_mask)
    V_pt = np.array(V_pt)
    V_ft = np.array(V_ft)
    V_dt = np.array(V_dt)
    V_num = np.array(V_num)
    V_dispatch_mask = np.array(V_dispatch_mask)
    start_idx = np.array(start_idx)
    E_ed = np.array(E_ed)
    E_sd = np.array(E_sd)
    pt_dif = np.array(pt_dif)
    dt_dif = np.array(dt_dif)
    label = np.array(label)
    cou = np.array(cou)
    E_mask = np.array(E_mask)
    A = np.array(A)
    index = np.array(index)
    E = np.zeros([label.shape[0], label.shape[1], E_ed.shape[1], E_ed.shape[2], 5])
    for t in range(label.shape[1]):
        E_mask_t = E_mask[:, t, :, :]
        E_ed_dif_t = (E_ed * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
        E_sd_dif_t = (E_sd * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
        E_pt_t = (pt_dif * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
        E_dt_t = (dt_dif * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
        A_t = (A[:, t, :, :] * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
        E_t = np.concatenate([E_ed_dif_t, E_sd_dif_t, E_pt_t, E_dt_t, A_t], axis=3)
        E[:, t, :, :, :] = E_t

    V = torch.FloatTensor(V).to(device)
    V_reach_mask = torch.BoolTensor(V_reach_mask).to(device)

    label = torch.LongTensor(label).to(device)

    V_pt = torch.FloatTensor(V_pt).to(device)
    V_ft = torch.FloatTensor(V_ft).to(device)
    start_idx = torch.LongTensor(start_idx).to(device)

    V_dt = torch.FloatTensor(V_dt).to(device)
    V_num = torch.FloatTensor(V_num).to(device)

    V_dispatch_mask = torch.FloatTensor(V_dispatch_mask).to(device)
    E_ed = torch.FloatTensor(E_ed).to(device)
    E_sd = torch.FloatTensor(E_sd).to(device)
    cou = torch.LongTensor(cou).to(device)
    E = torch.FloatTensor(E).to(device)
    index = torch.IntTensor(index).to(device)
    pred_scores, pred_pointers, b_V_val_unmasked = model.forward(V, V_reach_mask, V_ft, V_pt, V_dt, V_num,
                                                                 V_dispatch_mask, E, E_ed, E_sd, np.array(E_mask),
                                                                 start_idx, cou)

    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=pad_vaule)
    return pred_pointers, loss, b_V_val_unmasked, index


def test_model(modelRoute, test_dataloader, device, pad_value, params, save2fileRoute, mode, modelArrivalTime,
               save2fileArrivalTime):
    modelRoute.eval()

    evaluator_1 = Metric([params['eval_start'], params['eval_end_1']])
    evaluator_2 = Metric([params['eval_start'], params['eval_end_2']])

    total_pred_scores = torch.tensor([]).to(device)
    total_pred_pointers = torch.tensor([]).to(device)
    total_node_features = torch.tensor([]).to(device)
    total_index = torch.tensor([]).to(device)
    loss_batch = np.array([])

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            E_ed, V, V_reach_mask, label_len, label, V_pt, V_ft, start_idx, \
                E_sd, V_dt, V_num, E_mask, V_dispatch_mask, pt_dif, dt_dif, cou, A, last_x, last_len, unpick_x, unpick_len, \
                eta_np, order_np, index_np, index = zip(*batch)
            V = np.array(V)
            V_reach_mask = np.array(V_reach_mask)
            V_pt = np.array(V_pt)
            V_ft = np.array(V_ft)
            E_ed = np.array(E_ed)
            E_sd = np.array(E_sd)
            pt_dif = np.array(pt_dif)
            dt_dif = np.array(dt_dif)
            label = np.array(label)
            E_mask = np.array(E_mask)
            A = np.array(A)
            index = np.array(index)
            E = np.zeros([label.shape[0], label.shape[1], E_ed.shape[1], E_ed.shape[2], 5])
            for t in range(label.shape[1]):
                E_mask_t = E_mask[:, t, :, :]
                E_ed_dif_t = (E_ed * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
                E_sd_dif_t = (E_sd * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
                E_pt_t = (pt_dif * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
                E_dt_t = (dt_dif * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
                A_t = (A[:, t, :, :] * E_mask_t).reshape([E_ed.shape[0], E_ed.shape[1], E_ed.shape[2], 1])
                E_t = np.concatenate([E_ed_dif_t, E_sd_dif_t, E_pt_t, E_dt_t, A_t], axis=3)
                E[:, t, :, :, :] = E_t

            V = torch.FloatTensor(V).to(device)
            label_len = torch.LongTensor(label_len).to(device)
            label = torch.LongTensor(label).to(device)

            V_pt = torch.FloatTensor(V_pt).to(device)
            V_ft = torch.FloatTensor(V_ft).to(device)
            start_idx = torch.LongTensor(start_idx).to(device)

            V_dt = torch.FloatTensor(V_dt).to(device)
            V_num = torch.FloatTensor(V_num).to(device)

            V_dispatch_mask = torch.FloatTensor(V_dispatch_mask).to(device)
            V_reach_mask = torch.BoolTensor(V_reach_mask).to(device)
            E_ed = torch.FloatTensor(E_ed).to(device)
            E_sd = torch.FloatTensor(E_sd).to(device)
            cou = torch.LongTensor(cou).to(device)
            E = torch.FloatTensor(E).to(device)
            index = torch.IntTensor(index).to(device)
            """
            _KI_
            change the data type of ETPA parameters tuples->arrays

            """
     
            last_x = np.array(last_x)
            last_len = np.array(last_len)
            unpick_x = np.array(unpick_x)
            unpick_len = np.array(unpick_len)
            label_eta = np.array(eta_np)
            label_order = np.array(order_np)
            label_idx = np.array(index_np)

            """
            _KI_
            #  change the data type of ETPA parameters arrays->Tensors
            """
            last_x = torch.FloatTensor(last_x).to(device)
            last_len = torch.FloatTensor(last_len).to(device)
            unpick_x = torch.FloatTensor(unpick_x).to(device)
            unpick_len = torch.FloatTensor(unpick_len).to(device)
            label_eta = torch.FloatTensor(label_eta).to(device)
            label_order = torch.LongTensor(label_order).to(device)
            label_idx = torch.LongTensor(label_idx).to(device)
            """
            _KI_
            # Reshape input to B*T ...
            """
            _B, _T, _N = label_eta.shape
            last_x = last_x.reshape((_B * _T, _N, -1))
            last_len = last_len.reshape((_B * _T))
            unpick_x = unpick_x.reshape((_B * _T, _N, -1))
            unpick_len = unpick_len.reshape((_B * _T))
            label_eta = label_eta.reshape((_B * _T, _N))
            label_order = label_order.reshape((_B * _T, _N))
            label_idx = label_idx.reshape((_B * _T, _N))

            pred_scores, pred_pointers, b_V_val_unmasked = modelRoute(V, V_reach_mask, V_ft, V_pt, V_dt, V_num,
                                                                      V_dispatch_mask, E, E_ed, E_sd, np.array(E_mask),
                                                                      start_idx, cou)

            unrolled = pred_scores.view(-1, pred_scores.size(-1))
            loss_g2r = F.cross_entropy(unrolled, label.view(-1), ignore_index=params['pad_value'])
            """
            _KI_
            #  ETPA integration
            #  Code for pnn,
            # --------------------Before Masking ( Assign ranking to train and test )---------------------------------
            #  As the tensors inside dictionary were float -> first changed it to int values
            # --------------------After Masking ( Assign masked ranking to train and test )---------------------------------
             # Define the file name of the output file (.pkl)
            """
            pred = pred_pointers.to(torch.int32)  

            mask_value = torch.tensor(-1, dtype=torch.int32).to(device)
            mask = torch.eq(pred, 26) 
            masked_tensor = torch.where(mask, mask_value, pred)
            sort_idx = pred
            sort_pos = masked_tensor

            (input_idx, input_order) = (label_idx, label_order) if params['train_mode'] == 'true' else (
            sort_idx, sort_pos)

            pred_etpa, loss_etpa, n = modelArrivalTime(last_x, last_len, unpick_x, unpick_len, label_idx, label_order,
                                                       label_eta, input_idx, input_order)

            alpha = params['alpha_loss']
            joint_loss = alpha * loss_g2r + (1 - alpha) * (loss_etpa / 1000)
            loss_batch = np.append(loss_batch, joint_loss.cpu().detach().numpy())

            total_pred_scores = torch.cat((total_pred_scores, pred_scores), 0)
            total_pred_pointers = torch.cat((total_pred_pointers, pred_pointers), 0)
            total_node_features = torch.cat((total_node_features, b_V_val_unmasked), 0)
            total_index = torch.cat((total_index, index), 0)

            N = pred_pointers.size(-1)
            pred_len = torch.sum((pred_pointers.reshape(-1, N) < N - 1) + 0, dim=1)

            pred_steps, label_steps, labels_len, preds_len = \
                get_nonzeros(pred_pointers.reshape(-1, N), label.reshape(-1, N),
                             label_len.reshape(-1), pred_len, pad_value)

            evaluator_1.update(pred_steps, label_steps, labels_len, preds_len)
            evaluator_2.update(pred_steps, label_steps, labels_len, preds_len)

        if mode == 'val':
            return evaluator_2, pred_etpa, label_eta, np.mean(loss_batch)

        params_1 = dict_merge([evaluator_2.to_dict(), params])
        params_1['eval_min'] = params['eval_start']
        params_1['eval_max'] = params['eval_end_1']
        save2fileRoute(params_1)

        params_2 = dict_merge([evaluator_2.to_dict(), params])
        params_2['eval_min'] = params['eval_start']
        params_2['eval_max'] = params['eval_end_2']
        save2fileRoute(params_2)

       
        output_fname = f'route_result_{params["spatial_encoder"]}_{params["temporal_encoder"]}_{params["seed"]}.pkl'
        output_dict = {}

        if Path(output_fname).is_file():
            output_dict = np.load(output_fname, allow_pickle=True)

        output_dict['pred_pointers_test'] = total_pred_pointers

        with open(output_fname, 'wb') as df_file:
            pickle.dump(obj=output_dict, file=df_file)

        output_dict_node = {}
        output_dict_node['V_val'] = total_node_features
        output_fname_node = 'node_features_test.npy'
        with open(output_fname_node, 'wb') as df_file:
            pickle.dump(obj=output_dict_node, file=df_file)

        return evaluator_2, pred_etpa, label_eta


def main(params):
    run(params, Graph2RouteDataset, process_batch, test_model, collate_fn)


def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":

    import time, nni
    import logging

    logger = logging.getLogger('training')
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)

        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise