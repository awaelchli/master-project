from base import Logger

l = Logger('out.txt')

l.column('col1', '{:.2f}')
l.column('col2', '{:d}')

l.log(0.1235523, 12293)

