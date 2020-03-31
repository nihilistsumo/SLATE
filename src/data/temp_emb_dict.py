from sentbert_embed import SentbertParaEmbedding
import numpy as np
import json
import argparse

def create_temp_emb_dir(emb_dir, emb_file_prefix, emb_paraids_file, query_attn_data_file, outfile, batch_size=10000):
    all_ids = np.load(emb_paraids_file)
    all_id_part_dict = {}
    for i in range(all_ids.size):
        all_id_part_dict[all_ids[i]] = ((i // batch_size) + 1, i % batch_size)
    part_para_dict = {}

    p_list = set()
    pemb_dict = {}
    with open(query_attn_data_file, 'r') as qd:
        for l in qd:
            p_list.add(l.split('\t')[2])
            p_list.add(l.split('\t')[3].rstrip())
    print('Have to find embeddings of '+str(len(p_list))+' paras')
    for p in p_list:
        part = all_id_part_dict[p][0]
        if part in part_para_dict.keys():
            part_para_dict[part].append(p)
        else:
            part_para_dict[part] = [p]
    print('part para dict created')
    for pt in part_para_dict.keys():
        emb_vecs = np.load(emb_dir + '/' + emb_file_prefix + '-part' + str(pt) + '.npy')
        for para in part_para_dict[pt]:
            pemb_dict[para] = emb_vecs[all_id_part_dict[para][1]]
        print('.')

    with open(outfile, 'w') as out:
        json.dump(pemb_dict, out)

def main():
    parser = argparse.ArgumentParser(description='Create temp para emb dict')
    parser.add_argument('-ed', '--emb_dir', help='Path to emb dir')
    parser.add_argument('-ep', '--emb_paras', help='Path to emb paraids file')
    parser.add_argument('-pre', '--emb_prefix', help='Embedding file prefix')
    parser.add_argument('-qp', '--qry_attn_file', help='Path to query attn file')
    parser.add_argument('-b', '--batch', help='Batch size for each embedding shards')
    parser.add_argument('-out', '--outfile', help='Path to output file')
    args = vars(parser.parse_args())
    emb_dir = args['emb_dir']
    emb_paraids_file = args['emb_paras']
    prefix = args['emb_prefix']
    qa_file = args['qry_attn_file']
    batch = int(args['batch'])
    outfile = args['outfile']
    create_temp_emb_dir(emb_dir, prefix, emb_paraids_file, qa_file, outfile, batch)

if __name__ == '__main__':
    main()