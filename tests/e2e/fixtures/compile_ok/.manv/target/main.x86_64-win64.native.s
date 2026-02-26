# manv backend assembly target=x86_64-win64
.text

.intel_syntax noprefix

.globl main
main:
  .seh_proc main
  push rbp
  .seh_pushreg rbp
  mov rbp, rsp
  .seh_setframe rbp, 0
  sub rsp, 96
  .seh_stackalloc 96
  .seh_endprologue
  # win64 shadow space reserved
main_entry:
  mov r10, 2
  mov r11, 3
  mov r12, r10
  add r12, r11
  mov QWORD PTR [rbp-8], r12
  mov r13, QWORD PTR [rbp-8]
  mov rcx, r13
  call manv_rt_print_i64
  mov r14, rax
  mov r15, 0
  mov rax, r15
  mov rsp, rbp
  pop rbp
  ret
  .seh_endproc


# runtime helper stubs
.globl manv_rt_print_i64
manv_rt_print_i64:
  xor eax, eax
  ret