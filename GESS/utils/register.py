r"""A kernel module that contains a global register for unified models, datasets, and learning algorithms access.
"""

class Register(object):
    r"""
    Global register for unified piepelines, models, datasets, and algorithms access.
    """

    def __init__(self):
        self.pipelines = dict()
        self.gdlbackbones = dict()
        self.models = dict()
        self.datasets = dict()
        self.dataloaders = dict()
        self.algorithms = dict()

    def pipeline_register(self, pipeline_class):
        r"""
        Register for pipeline access.

        Args:
            pipeline_class (class): pipeline class

        Returns (class):
            pipeline class

        """
        self.pipelines[pipeline_class.__name__] = pipeline_class
        return pipeline_class

    def gdlbackbone_register(self, gdl_class):
        r"""
        Register for GDL encoder access.

        Args:
            gdl_class (class): model class

        Returns (class):
            GDL encoder class

        """
        self.gdlbackbones[gdl_class.__name__] = gdl_class
        return gdl_class
    
    def model_register(self, model_class):
        r"""
        Register for ML model access.

        Args:
            model_class (class): model class

        Returns (class):
            model class

        """
        self.models[model_class.__name__] = model_class
        return model_class

    def dataloader_register(self, dataloader_class):
        r"""
        Register for dataloader access.

        Args:
            dataloader_class (class): dataloader class

        Returns (class):
            dataloader class

        """
        self.dataloaders[dataloader_class.__name__] = dataloader_class
        return dataloader_class

    def dataset_register(self, dataset_class):
        r"""
        Register for dataset access.

        Args:
            dataset_class (class): dataset class

        Returns (class):
            dataset class

        """
        self.datasets[dataset_class.__name__] = dataset_class
        return dataset_class

    def algorithm_register(self, baseline_class):
        r"""
        Register for learning algorithms access.

        Args:
            baseline_class (class): learning baseline_class class

        Returns (class):
            learning algorithms class

        """
        self.algorithms[baseline_class.__name__] = baseline_class
        return baseline_class


register = Register()  #: The GeSS register object used for accessing GDL backbones, models, datasets and OOD algorithms.