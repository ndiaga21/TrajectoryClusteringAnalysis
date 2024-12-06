import unittest
import pandas as pd
from TCA import TCA

class TestTCA(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Month1': [1, 2, 1, 2],
            'Month2': [2, 1, 2, 1]
        })  
        self.state_mapping = {'State1': 1, 'State2': 2}
        self.tca = TCA(self.data, self.state_mapping, colors=['blue', 'red'])

    def test_init(self):
        self.assertEqual(self.tca.data.equals(self.data), True)
        self.assertEqual(self.tca.state_label, ['State1', 'State2'])
        self.assertEqual(self.tca.state_numeric, [1, 2])
        self.assertEqual(self.tca.colors, ['blue', 'red'])

        # Test de l'exception pour les couleurs
        with self.assertRaises(ValueError):
            TCA(self.data, self.state_mapping, colors=['blue'])

    def test_plot_treatment_percentages(self):
        # Test si la méthode crée le bon nombre de traces
        df_test = pd.DataFrame({
            'Month1': [1, 1, 2, 2],
            'Month2': [2, 2, 1, 1]
        })
        self.tca.plot_treatment_percentages(df_test)

if __name__ == '__main__':
    unittest.main()