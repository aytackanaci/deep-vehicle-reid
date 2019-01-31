from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil

from .iotools import mkdir_if_missing


def visualize_ranked_results(distmat, all_AP, dataset, dataset_name, save_path, topk=20):

    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_path: directory to save html file with ranks
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    save_html = True


    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_path))

    # HELPERS
    def _printHtml(querys, ranks, data_root, f):
        header = '''
        <html>
            <head>
                <style>
                    img { width: 112px; height: 112px; border: 3px solid black;}
                    img.hit { border: 4px solid green; }
                    img.not { border: 4px solid red; }
                    td { text-align: right;}
                </style>
            </head>
            <body>
        '''

        table_header = '''
         <table border="1" class="dataframe">
           <thead>
             <tr style="text-align: left;">
               <th>query no</th>
               <th>ap</th>
               <th>query</th>
               <th>ranks</th>
             </tr>
           </thead>

        '''

        im_template = '<img title="{}" class="{}" src="{}" />'

        footer = '''
            </body>
        </html>
        '''

        f.write(header)
        f.write(table_header)
        #write table
        f.write('<tbody>\n')
        for i, q in enumerate(query):
            f.write('<tr>\n')

            f.write('<td># {:03d}</td><td>{:.3f}</td>'.format(i,all_AP[i]))

            f.write('<td>')
            f.write(im_template.format(q[0], '', q[0]))
            f.write('</td>\n')

            f.write('<td>')
            for image, hit in ranks[i]:
                c = 'hit' if hit else 'not'
                f.write(im_template.format(image, c, image))
            f.write('</td>\n')

            f.write('</tr>\n')

        f.write('</tbody>\n')
        f.write(footer)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)
    # end HELPERS


    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_path)

    g_ranks = []
    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        # if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
        #     qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        # else:
        #     qdir = osp.join(save_dir, osp.basename(qimg_path))
        # mkdir_if_missing(qdir)
        # _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        hitOrNot = []
        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                # _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                hitOrNot.append( (gimg_path, qpid == gpid) )
                if rank_idx > topk:
                    g_ranks.append( hitOrNot )
                    break


    if save_html:
        assert len(query) == len(g_ranks)
        html_out = osp.join(save_path, '{}_ranks.html'.format(dataset_name))
        with open(html_out, 'w') as fout:
            _printHtml(query, g_ranks, '', fout)

    print("Done")

