from sentence_transformers import SentenceTransformer
import argparse
import numpy as np
import spacy
import sys

class SentbertParaEmbedding():
    def __init__(self, paraid_file, embed_dir, embed_file_prefix, batch_size):
        self.para_info = None
        self.paraids = list(np.load(paraid_file))
        if('\t' in self.paraids[0]):
            self.para_info = self.paraids
            self.paraids = [p.split('\t')[0] for p in self.paraids]
        self.emb_dir = embed_dir
        self.prefix = embed_file_prefix
        self.batch_size = batch_size
        self.curr_para_emb = None
        self.part = -1

    def get_embeddings_as_dict(self, paraid_list):
        print('Going to retrieve embeddings for ' + str(len(paraid_list)) + ' paras')
        emb_dict = dict()
        #part = -1
        for p in paraid_list:
            emb_dict[p] = self.get_single_embedding(p)
        return emb_dict

    def get_single_embedding(self, paraid):
        #part = -1
        if paraid in self.paraids:
            p_index = self.paraids.index(paraid)
        else:
            print(paraid + ' not found in embedding dir')
            return None
        curr_part = p_index // self.batch_size + 1
        offset = p_index % self.batch_size
        if curr_part == self.part:
            emb_vec = self.curr_para_emb[offset]
        else:
            self.curr_para_emb = np.load(self.emb_dir + '/' + self.prefix + '-part' + str(curr_part) + '.npy')
            emb_vec = self.curr_para_emb[offset]
            self.part = curr_part
        return emb_vec

    def get_single_sent_embedding(self, paraid):
        if paraid in self.paraids:
            p_index = self.paraids.index(paraid)
            p_part = int(self.para_info[p_index].split('\t')[1])
            p_offset = int(self.para_info[p_index].split('\t')[2])
            p_len = int(self.para_info[p_index].split('\t')[3])
        else:
            print(paraid + ' not found in embedding dir')
            return None
        curr_part = p_part
        if curr_part == self.part:
            emb_vec = []
            for i in range(p_len):
                emb_vec.append(self.curr_para_emb[p_offset + i])
        else:
            self.curr_para_emb = np.load(self.emb_dir + '/' + self.prefix + '-part' + str(curr_part) + '.npy')
            emb_vec = []
            for i in range(p_len):
                emb_vec.append(self.curr_para_emb[p_offset + i])
            self.part = curr_part
        return emb_vec

    def get_sent_embeddings_as_dict(self, paraid_list):
        emb_dict = dict()
        for pid in paraid_list:
            emb_dict[pid] = self.get_single_sent_embedding(pid)
        return emb_dict

# get embeddings of the whole para
def get_embeddings(paratext_file, is_model_address, model_name, outdir, saveid=False, batch_size=10000):
    model = SentenceTransformer(model_name)
    print("Using "+model_name+" to embed paras as a whole")
    with open(paratext_file, 'r') as pt:
        c = 0
        part = 0
        texts = []
        ids = []
        for l in pt:
            ids.append(l.split('\t')[0])
            if(len(l.split('\t')) < 2):
                texts.append(" ")
                print(l + 'No text found, appending space')
                sys.exit(0)
            else:
                texts.append(l.split('\t')[1])
            c += 1
            if c % batch_size == 0:
                print("Going to embed")
                embeds = model.encode(texts)
                part += 1
                print("Embedding complete, going to save part "+str(part)+", "+str(c)+" paras embedded\n")
                if is_model_address:
                    np.save(outdir + '/' + model_name.split('/')[len(model_name.split('/')) - 1] + '-part' + str(part),
                            embeds)
                else:
                    np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
                texts = []
        print("Going to embed")
        embeds = model.encode(texts)
        part += 1
        print("Embedding complete, going to save part " + str(part) + ", " + str(c) + " paras embedded\n")
        #np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
        if is_model_address:
            np.save(outdir + '/' + model_name.split('/')[len(model_name.split('/')) - 1] + '-part' + str(part),
                    embeds)
        else:
            np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
        if saveid:
            print("Saving paraids file")
            ids = np.array(ids)
            np.save(outdir + '/paraids', ids)

def get_sentence_wise_embeddings(paratext_file, is_model_address, model_name, outdir, saveid=False, batch_size=10000):
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')
    model = SentenceTransformer(model_name)
    print("Using " + model_name + " to embed paras sentence wise")
    with open(paratext_file, 'r') as pt:
        c = 0
        part = 0
        offset = 0
        texts = []
        ids = []
        for l in pt:
            paraid = l.split('\t')[0]
            text = l.split('\t')[1]
            text_sents = [str(s) for s in nlp(text).sents]
            text_len = len(text_sents)
            #text_sents = text.split('.')
            # ID part-no offset length
            ids.append(paraid + '\t' + str(part + 1) + '\t' + str(offset) + '\t' + str(text_len))
            #for i in range(len(text_sents)):
            #    ids.append(paraid+'_'+str(i+1))
            texts += text_sents
            c += 1
            offset += text_len
            if c % batch_size == 0:
                print("Going to embed")
                embeds = model.encode(texts)
                part += 1
                offset = 0
                print("Embedding complete, going to save part " + str(part) + ", " + str(c) + " paras embedded\n")
                if is_model_address:
                    np.save(outdir + '/' + model_name.split('/')[len(model_name.split('/')) - 1] + '-part' + str(part),
                            embeds)
                else:
                    np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
                texts = []
        if len(texts) > 0:
            print("Going to embed")
            embeds = model.encode(texts)
            part += 1
            print("Embedding complete, going to save part " + str(part) + ", " + str(c) + " paras embedded\n")
            #np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
            if is_model_address:
                np.save(outdir + '/' + model_name.split('/')[len(model_name.split('/')) - 1] + '-part' + str(part),
                        embeds)
            else:
                np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
        if saveid:
            print("Saving paraids file")
            ids = np.array(ids)
            np.save(outdir + '/paraids_sents', ids)

def main():
    parser = argparse.ArgumentParser(description='Use sentence-transformers to embed paragraphs')
    parser.add_argument('-p', '--paratext_file', help='Path to paratext file')
    parser.add_argument('-t', '--emb_type', help='s = sentence wise emb, p = whole para emb')
    parser.add_argument('-ma', '--model_address', action='store_true', help='Treat model name as model dir address')
    parser.add_argument('-m', '--model_name', help='Name of the model to be used for embedding')
    parser.add_argument('-o', '--outdir', help='Path to output dir')
    parser.add_argument('-i', '--save_id', type=bool, default=False, help='Save id file?')
    parser.add_argument('-b', '--batch_size', default=10000, help='Size of each batch to be encoded')
    args = vars(parser.parse_args())
    paratext_file = args['paratext_file']
    emb_type = args['emb_type']
    is_model_address = args['model_address']
    model_name = args['model_name']
    outdir = args['outdir']
    saveid = args['save_id']
    batch = int(args['batch_size'])
    if emb_type == 'p':
        get_embeddings(paratext_file, is_model_address, model_name, outdir, saveid, batch)
    elif emb_type == 's':
        get_sentence_wise_embeddings(paratext_file, is_model_address, model_name, outdir, saveid, batch)
    else:
        print("Wrong embedding type")

if __name__ == '__main__':
    main()