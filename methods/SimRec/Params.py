import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=4096, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
	parser.add_argument('--cdreg', default=1e-2, type=float, help='contrastive distillation reg weight, i.e. the embedding-level distillation')
	parser.add_argument('--softreg', default=1, type=float, help='soft-target-based distillation reg weight, i.e. the prediction-level distillation')
	parser.add_argument('--screg', default=1, type=float, help='weight for the contrastive regularization')
	parser.add_argument('--decay', default=1.0, type=float, help='regularization per-epoch decay')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=32, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--teacher_model', default=None, help='model name for teacher to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--topRange', default=100000, type=int, help='adaptive pick range')
	parser.add_argument('--tempsoft', default=0.03, type=float, help='temperature for prediction-level distillation')
	parser.add_argument('--tempcd', default=0.1, type=float, help='temperature for embedding-level distillation')
	parser.add_argument('--tempsc', default=1, type=float, help='temperature for contrastive regularization')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	parser.add_argument('--seed', default=None, type=int, help='random seed')
	return parser.parse_args()
args = ParseArgs()
