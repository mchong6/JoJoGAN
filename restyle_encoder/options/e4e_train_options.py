from options.train_options import TrainOptions


class e4eTrainOptions(TrainOptions):

    def __init__(self):
        super(e4eTrainOptions, self).__init__()

    def initialize(self):
        super(e4eTrainOptions, self).initialize()
        self.parser.add_argument('--w_discriminator_lambda', default=0, type=float,
                                 help='Dw loss multiplier')
        self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float,
                                 help='Dw learning rate')
        self.parser.add_argument("--r1", type=float, default=10,
                                 help="weight of the r1 regularization")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="interval for applying r1 regularization")
        self.parser.add_argument('--use_w_pool', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument("--w_pool_size", type=int, default=50,
                                 help="W\'s pool size, depends on --use_w_pool")

        # e4e_modules specific
        self.parser.add_argument('--delta_norm', type=int, default=2,
                                 help="norm type of the deltas")
        self.parser.add_argument('--delta_norm_lambda', type=float, default=2e-4,
                                 help="lambda for delta norm loss")

        # Progressive training
        self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
        self.parser.add_argument('--progressive_start', type=int, default=None,
                                 help="The training step to start training the deltas, overrides progressive_steps")
        self.parser.add_argument('--progressive_step_every', type=int, default=2_000,
                                 help="Amount of training steps for each progressive step")

        # Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument('--sub_exp_dir', default=None, type=str,
                                 help='Name of sub experiment directory')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")

    def parse(self):
        opts = self.parser.parse_args()
        return opts
