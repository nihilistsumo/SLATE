from sentbert_embed import SentbertParaEmbedding
import json
import argparse

def create_temp_emb_dir(emb_dir, emb_file_prefix, emb_paraids_file, query_attn_data_file, outfile, batch_size=10000):
    emb = SentbertParaEmbedding(emb_paraids_file, emb_dir, emb_file_prefix, batch_size)
    p_list = set()
    pemb_dict = {}
    with open(query_attn_data_file, 'r') as qd:
        for l in qd:
            p_list.add(l.split('\t')[2])
            p_list.add(l.split('\t')[3].rstrip())
    print('Have to find embeddings of '+str(len(p_list))+' paras')
    c = 0
    for p in p_list:
        pemb_dict[p] = emb.get_single_embedding(p)
        c += 1
        if c%1000 == 0:
            print(str(c)+' paras completed')
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