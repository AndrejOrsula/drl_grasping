#!/usr/bin/env python3

import argparse
from drl_grasping.utils import ModelCollectionRandomizer


def main(args=None):

    model_collection_randomizer = ModelCollectionRandomizer(owner=args.owner,
                                                            collection=args.collection,
                                                            server=args.server,
                                                            server_version=args.version,
                                                            unique_cache=True,
                                                            enable_blacklisting=True)
    print("Processing all models from owner [%s]..." % args.owner)
    model_collection_randomizer.process_all_models(decimation_fraction_of_visual=0.025,
                                                   decimation_min_faces=8,
                                                   decimation_max_faces=400,
                                                   max_faces=40000,
                                                   max_vertices=None,
                                                   component_min_faces_fraction=0.05,
                                                   component_max_volume_fraction=0.1,
                                                   fix_mtl_texture_paths=True)
    print("Processing finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Process all models from a collection. ' +
                    'If local cache already contains one or more models from the owner [-o], ' +
                    'these models will be processed and [-s, -c, -v] arguments ignored.')
    parser.add_argument('-o', '--owner',
                        action='store',
                        default='GoogleResearch',
                        help='Owner of the collection')
    parser.add_argument('-c', '--collection',
                        action='store',
                        default='Google Scanned Objects',
                        help='Name of the collection')
    parser.add_argument('-s', '--server',
                        action='store',
                        default='https://fuel.ignitionrobotics.org',
                        help='URI to Ignition Fuel server where the collection resides')
    parser.add_argument('-v', '--version',
                        action='store',
                        default='1.0',
                        help='Version of the Fuel server')

    args = parser.parse_args()

    main(args)
