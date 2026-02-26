# host stubs for x86_64-sysv
.globl launch_main_kernel_0
launch_main_kernel_0:
  # ABI-compliant host launcher stub
  # marshal pointers/scalars to runtime launch API
  # kernel=main_kernel_0
  # params=4 sync=default
  xor eax, eax
  ret
