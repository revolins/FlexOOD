import pytest


def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', help='run tests that need a CUDA device')
    parser.addoption('--slow', action='store_true', help='run the slow CPU tests (ogbl-collab AA-DC)')


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: needs a CUDA device (--gpu)')
    config.addinivalue_line('markers', 'slow: slow CPU test (--slow)')


def pytest_collection_modifyitems(config, items):
    for item in items:
        if 'gpu' in item.keywords and not config.getoption('--gpu'):
            item.add_marker(pytest.mark.skip(reason='needs --gpu'))
        if 'slow' in item.keywords and not config.getoption('--slow'):
            item.add_marker(pytest.mark.skip(reason='needs --slow'))
