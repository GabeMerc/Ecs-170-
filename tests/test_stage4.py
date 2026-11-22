import tempfile
import unittest
from pathlib import Path
import torch
from local_code.stage_4_code.Dataset_Loader import (Dataset_Loader, Vocabulary, tokenize, collate_reviews,
                                                    tokenize_joke, detokenize, JokeDataset, collate_jokes)
from local_code.stage_4_code.Evaluate_Text import TrainingCorpusIndex
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator

class Stage4Tests(unittest.TestCase):
    def test_cleaning(self):
        self.assertEqual(tokenize("<b>NOT</b> bad &amp; isn't dull!"), ['not','bad',"isn't",'dull'])
        vocab = Vocabulary(['good good movie'], min_freq=1)
        self.assertEqual(vocab.encode('unseen'), [1])
        self.assertEqual(vocab.encode('<br>'), [1])
        self.assertEqual(len(vocab.encode('good movie', 1)), 1)

    def test_loader(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ['train','test']:
                for label in ['neg','pos']:
                    p = Path(root)/'text_classification'/split/label
                    p.mkdir(parents=True)
                    (p/'1.txt').write_text('training word' if split=='train' else 'heldoutonly')
            p = Path(root)/'text_generation'
            p.mkdir()
            (p/'data').write_text('"ID","Joke"\n1,"Hello, world"\n')
            loader = Dataset_Loader(data_root=root, min_freq=1)
            data = loader.load()
            self.assertNotIn('heldoutonly', data['vocabulary'].word_to_id)
            self.assertEqual(data['test'][0][0].tolist(), [1])
            self.assertEqual(loader.load_generation(), ['Hello, world'])
            self.assertEqual(loader.inspect()['train']['count'], 2)

    def test_recurrent_models(self):
        for cell in ['rnn','lstm','gru']:
            for bidirectional in [False,True]:
                with self.subTest(cell=cell, bidirectional=bidirectional):
                    torch.manual_seed(21)
                    model = Method_RNN(vocab_size=10,embedding_dim=5,hidden_dim=7,
                                       num_layers=2,dropout=0,cell=cell,bidirectional=bidirectional)
                    tokens,lengths,labels = collate_reviews([
                        (torch.tensor([2,3,4]),torch.tensor(1)),
                        (torch.tensor([5]),torch.tensor(0))])
                    logits = model(tokens,lengths)
                    self.assertEqual(tuple(logits.shape),(2,2))
                    torch.testing.assert_close(logits[1],model(tokens[1:2,:1],lengths[1:2])[0])
                    torch.testing.assert_close(logits,model(torch.nn.functional.pad(tokens,(0,4)),lengths))
                    torch.nn.functional.cross_entropy(logits,labels).backward()
                    self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
                    self.assertEqual(model.embedding.weight.grad[0].abs().sum(),0)
                    self.assertEqual(tuple(model.predict(tokens,lengths).shape),(2,))

    def test_pretrained(self):
        model = Method_RNN(vocab_size=5,embedding_dim=3,pretrained_embeddings=torch.ones(5,3),freeze_embeddings=True)
        self.assertFalse(model.embedding.weight.requires_grad)
        self.assertEqual(model.embedding.weight[0].sum(),0)
        with self.assertRaises(ValueError): model(torch.tensor([[2]]),torch.tensor([0]))

    def test_validation_split(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ['train','test']:
                for label in ['neg','pos']:
                    p = Path(root)/'text_classification'/split/label
                    p.mkdir(parents=True)
                    for i in range(10):
                        (p/f'{i}.txt').write_text(f'{label}{i}')
            data = Dataset_Loader(data_root=root, min_freq=1).load(val_fraction=0.2, seed=3)
            self.assertEqual((len(data['train']),len(data['val'])),(16,4))
            self.assertEqual(data['val'].labels.sum().item(),2)  # stratified
            # Validation-only words must be unknown to the vocabulary.
            self.assertEqual(sum((seq.tolist() for seq in data['val'].sequences),[]),[1,1,1,1])

    def test_joke_pipeline(self):
        self.assertEqual(tokenize_joke("Why? It's \"wurst\"!"),['why','?',"it's",'"','wurst','"','!'])
        self.assertEqual(tokenize_joke('Wait... no.'),['wait','...','no','.'])
        self.assertEqual(detokenize(['hi',',','there','...','!']),'hi, there...!')
        vocab = Vocabulary(['a b.', 'a c!'], min_freq=1, tokenizer=tokenize_joke, specials=('<eos>',))
        self.assertEqual(vocab.tokens[:3],['<pad>','<unk>','<eos>'])
        inputs,lengths,targets = collate_jokes([JokeDataset(['a b.','a'],vocab)[i] for i in range(2)])
        self.assertEqual(lengths.tolist(),[3,1])
        self.assertEqual(vocab.decode(targets[0].tolist()),['b','.','<eos>'])
        self.assertEqual(targets[1].tolist(),[2,0,0])

    def test_generator(self):
        torch.manual_seed(21)
        model = Method_RNN_Generator(vocab_size=9,embedding_dim=4,hidden_dim=6,dropout=0)
        tokens = torch.tensor([[3,4,5],[6,0,0]])
        logits,_ = model(tokens,torch.tensor([3,1]))
        self.assertEqual(tuple(logits.shape),(2,3,9))
        torch.testing.assert_close(logits[1,0],model(tokens[1:2,:1])[0][0,0])
        model.train()
        greedy = model.generate([3,4,5],eos_id=2,max_new_tokens=10,temperature=0)
        self.assertTrue(model.training)
        self.assertEqual(greedy,model.generate([3,4,5],eos_id=2,max_new_tokens=10,temperature=0))
        self.assertLessEqual(len(greedy),10)
        self.assertFalse({0,1,2} & set(greedy))
        sample = model.generate([3],eos_id=2,top_k=2,generator=torch.Generator().manual_seed(1))
        self.assertFalse({0,1} & set(sample))

    def test_corpus_comparison(self):
        index = TrainingCorpusIndex([['what','did','the','cow','say','?'],['why','did','the','hen','run','?']])
        copy = index.compare(['what','did','the'],['what','did','the','cow','say','?'])
        self.assertTrue(copy['exact_copy'])
        self.assertEqual((copy['prefix_starts_jokes'],copy['longest_copied_span'],copy['3gram_in_training']),(1,6,1.0))
        novel = index.compare(['what','did','the'],['what','did','the','hen','run','?'])
        self.assertFalse(novel['exact_copy'])
        self.assertEqual(novel['longest_copied_span'],5)
        self.assertEqual(novel['prefix_appears_in_jokes'],1)

if __name__=='__main__': unittest.main()
