from msgflux.runtime.permissions import (
    PermissionSet,
    ResourcePermission,
    intersect_permissions,
)


def test_broad_parent_admits_exact_child_resource_in_active_workspace():
    path = ResourcePermission("workspace:project:/src/main.py", "filesystem.read")

    effective = intersect_permissions(
        PermissionSet({"filesystem.read", "catalog.query"}),
        PermissionSet(
            {"filesystem.read", "catalog.query"},
            {path},
        ),
        workspace_id="project",
    )

    assert effective.grants == {"filesystem.read", "catalog.query"}
    assert effective.resources == {path}


def test_exact_parent_resource_admits_broad_child_for_same_workspace():
    path = ResourcePermission("workspace:project:/src/main.py", "filesystem.read")

    effective = intersect_permissions(
        PermissionSet({"filesystem.read"}, {path}),
        PermissionSet({"filesystem.read"}),
        workspace_id="project",
    )

    assert effective.resources == {path}


def test_broad_grants_do_not_expand_other_workspace_or_external_resources():
    current_path = ResourcePermission(
        "workspace:project:/src/main.py", "filesystem.read"
    )
    other_path = ResourcePermission("workspace:other:/src/main.py", "filesystem.read")
    catalog = ResourcePermission("catalog:public", "catalog.read")

    effective = intersect_permissions(
        PermissionSet(
            {"filesystem.read"},
            {other_path, catalog},
        ),
        PermissionSet({"filesystem.read"}, {current_path, catalog}),
        workspace_id="project",
    )

    assert effective.resources == {catalog, current_path}
    assert other_path not in effective.resources


def test_workspace_resource_actions_must_match_the_broad_grant():
    process_workspace = ResourcePermission("workspace:project:/", "process.workspace")

    effective = intersect_permissions(
        PermissionSet({"process.execute"}),
        PermissionSet({"process.execute"}, {process_workspace}),
        workspace_id="project",
    )

    assert effective.resources == set()


def test_no_active_workspace_keeps_only_exact_resource_intersections():
    shared = ResourcePermission("workspace:project:/private", "filesystem.read")
    child_only = ResourcePermission("workspace:project:/public", "filesystem.read")

    effective = intersect_permissions(
        PermissionSet({"filesystem.read"}, {shared}),
        PermissionSet({"filesystem.read"}, {shared, child_only}),
    )

    assert effective.resources == {shared}

    broad_only = intersect_permissions(
        PermissionSet({"filesystem.read"}),
        PermissionSet({"filesystem.read"}, {child_only}),
    )
    assert broad_only.resources == set()
