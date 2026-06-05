"""
stage2_crossmodal.py — Stage 2: 跨模态 Hebb 学习
自由能原理智能体

用法: python stage2_crossmodal.py --dataset coco --n 5000 --mode all
实际代码位置: cerebrum/association/crossmodal.py
"""
if __name__ == '__main__':
    import sys
    from runpy import run_module
    sys.argv[0] = 'cerebrum.association.crossmodal'
    run_module('cerebrum.association.crossmodal', run_name='__main__')
