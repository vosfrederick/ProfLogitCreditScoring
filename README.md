# ProfLogitCreditScoring
Profit Maximizing Logistic Model for Credit Scoring

The classifier, named ProfLogit,
maximizes the EMPCS in the training step using a genetic algorithm, where
ProfLogit's interior model structure resembles a lasso-regularized logistic model.

This code is written on top of the code already existing which was written by E. Stripling (https://github.com/estripling/proflogit) with the purpose of applying the ProfLogit model on Credit Scoring. The EMP for churn is adapted to the EMP for Credit Scoring and by consequence a binary classification of clients is done here based on whether they default or not.

The accompanied paper entitled "*Profit Maximizing Logistic Model for Customer
Churn Prediction Using Genetic Algorithms*"
[is published](http://authors.elsevier.com/sd/article/S2210650216301754) in the international peer-reviewed journal of
[Swarm and Evolutionary Computation](https://www.journals.elsevier.com/swarm-and-evolutionary-computation).


Citation
--------

If you find ProfLogit useful, please cite it in your publications.
You can use the following [BibTeX](http://www.bibtex.org/) entry:

```
@article{stripling2018proflogit,
  title={{Profit Maximizing Logistic Model for Customer Churn Prediction Using Genetic Algorithms}},
  author={Stripling, Eugen and vanden Broucke, Seppe and Antonio, Katrien and Baesens, Bart and Snoeck, Monique},
  journal={Swarm and Evolutionary Computation},
  volume={40},
  pages={116-130},
  year={2018},
  issn={2210-6502},
  publisher={Elsevier},
  keywords={Data mining, Customer churn prediction, Lasso-regularized logistic regression model, Profit-based model evaluation, Real-coded genetic algorithm},
  doi={10.1016/j.swevo.2017.10.010},
}
```

License
-------
The code in this repository, including all code samples in the accompanied
notebooks, is released under the GNU General Public License v3 (GPLv3).

