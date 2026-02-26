# manv backend assembly target=x86_64-sysv
.text

.intel_syntax noprefix

.globl main
.type main, @function
main:
  .cfi_startproc
  push rbp
  .cfi_def_cfa_offset 16
  .cfi_offset rbp, -16
  mov rbp, rsp
  .cfi_def_cfa_register rbp
  sub rsp, 64
  .cfi_adjust_cfa_offset 64
main_entry:
  mov r10, 10
  mov QWORD PTR [rbp-8], r10
  mov r11, QWORD PTR [rbp-8]
  mov r12, 10
  mov r13, r11
  add r13, r12
  mov QWORD PTR [rbp-8], r13
  mov r14, QWORD PTR [rbp-8]
  mov rdi, r14
  call manv_rt_print_i64
  mov r15, rax
  mov rbx, QWORD PTR [rbp-8]
  mov rax, rbx
  mov rsp, rbp
  pop rbp
  ret
  .cfi_endproc


# runtime helper stubs
.globl manv_rt_print_i64
manv_rt_print_i64:
  xor eax, eax
  ret