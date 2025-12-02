import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eagletn7/Downloads/project-GO2/install/go2_simple_navigation'
