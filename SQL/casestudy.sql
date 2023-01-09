-- Show folder and file changes from initial commit 8be4d8a73b2d327687c053eeccd04551db5d9818

-- case study https://github.com/ClusterHQ/flocker/commits/6d79e2718f071d9189a21f0bda91cbdbd092a69d

select revision.id as revision_id, committer_date,  directory as directory_id, dir_entries, file_entries, rev_entries
from revision, directory
where revision.directory = directory.id
and revision.id = '\x8be4d8a73b2d327687c053eeccd04551db5d9818';

--                revision_id                 |     committer_date     |                directory_id                | dir_entries |   file_entries    | rev_entries
--------------------------------------------+------------------------+--------------------------------------------+-------------+-------------------+-------------
-- \x8be4d8a73b2d327687c053eeccd04551db5d9818 | 2014-04-28 20:02:14+02 | \x4ebc57d159631ad615c60fdbe2b1c195c61d2b75 |             | {29390,103255041} |


select revision.id as revision_id, committer_date,  directory as directory_id, dir_entries, file_entries, rev_entries
from revision, directory
where revision.directory = directory.id
and revision.id = '\x67a02b0bc03e276c458db7d39bb736fa5f6a6a1b';

--                revision_id                 |     committer_date     |                directory_id                | dir_entries |        file_entries         | rev_entries
--------------------------------------------+------------------------+--------------------------------------------+-------------+-----------------------------+-------------
-- \x67a02b0bc03e276c458db7d39bb736fa5f6a6a1b | 2014-04-28 20:44:20+02 | \x4b3a94889c88626e080e6096a6bd50d55f7a12e2 |             | {29390,103256229,103255041} |


select revision.id as revision_id, committer_date,  directory as directory_id, dir_entries, file_entries, rev_entries
from revision, directory
where revision.directory = directory.id
and revision.id = '\xb086670d6a0f2a30b562fe4142e0cb82aec0d68e';

-- directory_entry_dir_entry_id |                directory_id                | dir_entries | file_entries | rev_entries
------------------------------+--------------------------------------------+-------------+--------------+-------------
--                     82275439 | \x186d3cc219c0710e8a8d2e353e45fc59d84b3223 | {82258782}  | {103238560}  |

select revision.id as revision_id, committer_date,  directory as directory_id, dir_entries, file_entries, rev_entries
from revision, directory
where revision.directory = directory.id
and revision.id = '\x98510addd167be032a2fbc557c382c6227962f1b';
