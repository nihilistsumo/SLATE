from sentbert_embed import SentbertParaEmbedding
import numpy as np
import json
import argparse

def merge_parts(parts, prefix, emb_dir):
    ids = []
    embs = []
    all_ids = []
    all_embs = []
    for p in parts:
        ids.append(np.load(emb_dir+'/paraids_'+prefix+'-sents-part'+str(p)+'.npy'))
        embs.append(np.load(emb_dir+'/'+prefix+'-part'+str(p)+'.npy'))
    offsets = [0]
    count = 0
    for i in range(1, len(ids)):
        count += len(ids[i-1])
        offsets.append(count)
    for i in range(len(ids)):
        for id in ids[i]:
            all_ids.append(id.split('\t')[0] + '\t1\t' + str(offsets[i] + int(id.split('\t')[2])) + '\t' + id.split('\t')[3])
        if i == 0:
            all_embs = embs[i]
        else:
            all_embs = np.vstack((all_embs, embs[i]))
    return all_ids, all_embs

def create_temp_single_emb_dir(emb_dir, emb_file_prefix, emb_paraids_file, bert_seq_data_file, outdir, outfile, part_range):
    all_ids = np.load(emb_paraids_file)
    all_id_part_dict = {}
    for i in range(all_ids.size):
        all_id_part_dict[all_ids[i].split('\t')[0]] = (int(all_ids[i].split('\t')[1]), int(all_ids[i].split('\t')[2]),
                                                       int(all_ids[i].split('\t')[3]))
    p_list = set()
    with open(bert_seq_data_file, 'r') as qd:
        first = True
        for l in qd:
            if first:
                first = False
                continue
            p_list.add(l.split('\t')[1])
            p_list.add(l.split('\t')[2])
    part_para_dict = {}
    for p in p_list:
        part = all_id_part_dict[p][0]
        if part in part_para_dict.keys():
            part_para_dict[part].append(p)
        else:
            part_para_dict[part] = [p]
    print('part para dict created')
    print(str(part_range[1] - part_range[0])+' individual emb files to be used')
    para_num = sum([len(part_para_dict[p]) for p in part_para_dict.keys()])
    print('Have to find embeddings of ' + str(para_num) + ' paras')
    i = 0
    j = 0
    paras = []
    embs = []
    # for pt in part_para_dict.keys():
    for pt in range(part_range[0], part_range[1]):
        emb_vecs = np.load(emb_dir + '/' + emb_file_prefix + '-part' + str(pt) + '.npy')
        for para in part_para_dict[pt]:
            embvec = emb_vecs[all_id_part_dict[para][1]:all_id_part_dict[para][1] + all_id_part_dict[para][2]]
            paras.append(para + '\t1\t' + str(i) + '\t' + str(all_id_part_dict[para][2]))
            i += all_id_part_dict[para][2]
            for e in embvec:
                embs.append(e)
        j += 1
        print(j)

    np.save(outdir + '/' + outfile + '-part'+str(part_range[0])+'.npy', np.array(embs))
    np.save(outdir + '/paraids_' + outfile + '-sents-part'+str(part_range[0])+'.npy', np.array(paras))

def create_temp_emb_dir(emb_dir, emb_file_prefix, emb_paraids_file, bert_seq_data_file, outdir, outfile, batch_size=10000):
    all_ids = np.load(emb_paraids_file)
    all_id_part_dict = {}
    for i in range(all_ids.size):
        all_id_part_dict[all_ids[i].split('\t')[0]] = (int(all_ids[i].split('\t')[1]), int(all_ids[i].split('\t')[2]),
                                                       int(all_ids[i].split('\t')[3]))
    part_para_dict = {}

    p_list = set()
    with open(bert_seq_data_file, 'r') as qd:
        first = True
        for l in qd:
            if first:
                first = False
                continue
            p_list.add(l.split('\t')[1])
            p_list.add(l.split('\t')[2])
    print('Have to find embeddings of '+str(len(p_list))+' paras')
    for p in p_list:
        part = all_id_part_dict[p][0]
        if part in part_para_dict.keys():
            part_para_dict[part].append(p)
        else:
            part_para_dict[part] = [p]
    print('part para dict created')
    print(str(len(part_para_dict))+' individual emb files to be used')
    i = 0
    j = 0
    p = 0
    part = 1
    paras = []
    embs = []
    for pt in part_para_dict.keys():
        emb_vecs = np.load(emb_dir + '/' + emb_file_prefix + '-part' + str(pt) + '.npy')
        for para in part_para_dict[pt]:
            embvec = emb_vecs[all_id_part_dict[para][1]:all_id_part_dict[para][1]+all_id_part_dict[para][2]]
            paras.append(para+'\t'+str(part)+'\t'+str(i)+'\t'+str(all_id_part_dict[para][2]))
            i += all_id_part_dict[para][2]
            for e in embvec:
                embs.append(e)
            p += 1
            if p > 9999:
                np.save(outdir + '/' + outfile + '-part'+str(part)+'.npy', np.array(embs))
                np.save(outdir + '/paraids_' + outfile + '-part'+str(part)+'.npy', np.array(paras))
                part += 1
                i = 0
                p = 0
                paras = []
                embs = []
        j += 1
        print(j)

    np.save(outdir + '/' + outfile + '-part' + str(part) + '.npy', np.array(embs))
    np.save(outdir + '/paraids_' + outfile + '-part' + str(part) + '.npy', np.array(paras))

def main():
    parser = argparse.ArgumentParser(description='Create temp para emb dict')
    parser.add_argument('-ed', '--emb_dir', help='Path to emb dir')
    parser.add_argument('-ep', '--emb_paras', help='Path to emb paraids file')
    parser.add_argument('-pre', '--emb_prefix', help='Embedding file prefix')
    parser.add_argument('-bt', '--bert_seq_file', help='Path to input file in bert seq format')
    parser.add_argument('-b', '--batch', help='Batch size for each input embedding shards')
    parser.add_argument('-od', '--outdir', help='Path to output dir')
    parser.add_argument('-of', '--outfile', help='output temp emb file name without .npy')
    parser.add_argument('-pr', '--part_range', help='Part range in the form start-stop')
    args = vars(parser.parse_args())
    emb_dir = args['emb_dir']
    emb_paraids_file = args['emb_paras']
    prefix = args['emb_prefix']
    bt_file = args['bert_seq_file']
    batch = int(args['batch'])
    outdir = args['outdir']
    outfile = args['outfile']
    part_range = [int(args['part_range'].split('-')[0]), int(args['part_range'].split('-')[1])]
    create_temp_single_emb_dir(emb_dir, prefix, emb_paraids_file, bt_file, outdir, outfile, part_range)

if __name__ == '__main__':
    main()