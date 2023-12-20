"""
Test to see if the decoder class is functioning correctly
"""
from preferendus._decoder import _Decoding


def test_decoder():
    bounds = ((0, 7000), (0, 7000), (0, 1))
    n_bits = 64  # the higher, the more accurate, necessary for this test
    type_of_variables = ["real", "int", "bool"]

    cls = _Decoding(bounds=bounds, n_bits=n_bits, approach=type_of_variables)

    ip = [6875.54, 5000, 1]
    bit_strings = cls.inverse_decode(ip)
    decoded = cls.decode(bit_strings)
    assert ip == decoded
    return


if __name__ == "__main__":
    test_decoder()
