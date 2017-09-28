import os
import shutil
import tempfile
import tensorflow as tf


def export_model(session: tf.Session) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = os.path.join(tmpdir, 'export')
        os.makedirs(export_dir)
        export_path = os.path.join(export_dir, 'vars')
        export_archive = os.path.join(tmpdir, 'export.zip')
        saver = tf.train.Saver()
        saver.save(session, export_path)
        shutil.make_archive(export_archive.rsplit('.', 1)[0], 'zip', export_dir)
        with open(export_archive, 'rb') as f:
            return f.read()


def import_model(session: tf.Session, exported, restore_graph=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = os.path.join(tmpdir, 'export')
        export_archive = os.path.join(tmpdir, 'export.zip')
        export_path = os.path.join(export_dir, 'vars')
        with open(export_archive, 'wb') as tmpfile:
            tmpfile.write(exported)
        shutil.unpack_archive(export_archive, export_dir, 'zip')
        if restore_graph:
            saver = tf.train.import_meta_graph(export_path + '.meta')
        else:
            saver = tf.train.Saver()
        saver.restore(session, export_path)
