import unittest

from utils.cli_parser import parse_arguments


class CliParserTests(unittest.TestCase):
    def test_safe_mode_flags(self):
        args = parse_arguments(["--no-solution", "--display-coords"])

        self.assertFalse(args.drawSolution)
        self.assertTrue(args.displaySolutionCoords)

    def test_simulation_length(self):
        args = parse_arguments(["--sim-length", "250"])

        self.assertEqual(args.simulationLength, 250)


if __name__ == "__main__":
    unittest.main()
