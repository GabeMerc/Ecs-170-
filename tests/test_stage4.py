import tempfile
import unittest
from pathlib import Path
import torch
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, Vocabulary, tokenize, collate_reviews
from local_code.stage_4_code.Method_RNN import Method_RNN

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

if __name__=='__main__': unittest.main()
