=============================
Medical Concept Normalization
=============================

Medical Concept Normalization aims to map a medical phrase to a concept in a medical lexicon. In this work, we handle MCN as a two staged process. In the first stage, embeddings are learnt corresponding to each medical concept in a medical KB. In the second stage, a transformer based neural model is used to project an input phrase into the target embedding space. Finally, cosine similarity is used to find the nearest medical concept corresponding input phrase.


Encoding Target Knowledge
=========================

In order to encode the medical knowledge we learn an embedding corresponding to every medical concept in a medical lexicon or KB. For the extent of this work we use SNOMED CT International as the medical lexicon. We experiment with three different approaches to encode the medical concepts into embeddings. 

In the first approach we pass the concept description through a pretrained model and use the output of the model as the concept embedding. We have experimented with the following pretrained text encoders.

   * Averaged Glove Embeddings
   * Universal Sentence Encoder
   * ELMo
   * BERT

In the second approach, we construct a SNOMED CT concept graph where medical concepts are treated as vertices and similar related concepts are connected with an edge. We train the following graph embedding algorithm on this graph and obtain the embedding corresponding to vertices (or medical concepts)

   * Deepwalk
   * Node2Vec
   * HARP
   * LINE

In the third approach we pose SNOMED CT as a knowledge graph, i.e. as a collection of triplets. Each triplet has the following form *source_entity* *relation* *target_entity*. We use existing knowledge graph embedding methods discussed in `Pykg2Vec <https://github.com/Sujit-O/pykg2vec>`_ repository to encode medical concepts. In the following we explain the steps needed to obtain the concept embeddings.

   * Install the Pykg2Vec package using `python setup.py install`
   * Copy the `resources/snomed_kg` folder in the main Pykg2vec main directory.
   * Execute the command `python examples/train.py -h` to know all the options.
   * Execute the command `python examples/train.py -device cuda -npg <n_threads> -mn TransD -ds snomed -dsp snomed_kg/`
   * This will create folders inside the `resources/snomed_kg` folder which will contain all the results.


Pretrained Medical Concept Embeddings
=====================================

All the pretrained concept embeddings are freely accessible at https://zenodo.org/record/3842143


Credit and Citation
===================

Please kindly consider citing the following papers if you find this repository or the pretrained medical concept embeddings useful for your research.

.. code-block:: code

   @inproceedings{pattisapu2020medical,
     title={Medical Concept Normalization by Encoding Target Knowledge},
     author={Pattisapu, Nikhil and Patil, Sangameshwar and Palshikar, Girish and Varma, Vasudeva},
     booktitle={Machine Learning for Health Workshop},
     pages={246--259},
     year={2020}
   }
   
   @article{pattisapu2020distant,
     title={Distant Supervision for Medical Concept Normalization},
     author={Pattisapu, Nikhil and Anand, Vivek and Patil, Sangameshwar and Palshikar, Girish and Varma, Vasudeva},
     journal={Journal of biomedical informatics},
     year={2020},
     publisher={Elsevier}
   }
