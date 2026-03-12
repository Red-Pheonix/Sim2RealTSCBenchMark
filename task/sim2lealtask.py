import logging
from .task import BaseTask
from common.registry import Registry



@Registry.register_task("sim2real_transitions")
class Sim2RealTransitionsTask(BaseTask):
    '''
    Register Traffic Signal Control task.
    '''
    def run(self):
        '''
        run
        Run the whole task, including training and testing.

        :param: None
        :return: None
        '''
        try:
            if Registry.mapping['model_mapping']['setting'].param['train_model']:
                print("-----conducting training--------")
                self.trainer.train_flow()

            if Registry.mapping['model_mapping']['setting'].param['test_model']:
                print("-----conducting testing--------")
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e
