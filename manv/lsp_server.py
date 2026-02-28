from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse

from lsprotocol.types import (
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionOptions,
    DefinitionParams,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidCloseTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    DocumentSymbol,
    DocumentSymbolParams,
    Hover,
    HoverParams,
    InitializeParams,
    InitializeResult,
    Location,
    MarkupContent,
    MarkupKind,
    ParameterInformation,
    Position,
    PrepareRenameParams,
    Range,
    ReferenceParams,
    RenameParams,
    SemanticTokens,
    SemanticTokensLegend,
    SemanticTokensOptions,
    SemanticTokensParams,
    ServerCapabilities,
    SignatureHelp,
    SignatureHelpOptions,
    SignatureInformation,
    SymbolInformation,
    SymbolKind,
    TextDocumentSyncKind,
    TextEdit,
    WorkspaceEdit,
    WorkspaceSymbolParams,
)
from pygls.server import LanguageServer

from . import ast
from .compiler import parse_program
from .diagnostics import Diagnostic as ManvDiagnostic
from .diagnostics import ManvError
from .intrinsics import BUILTIN_ALIASES, all_intrinsics, resolve_intrinsic
from .lexer import Lexer
from .semantics import (
    SemanticAnalyzer,
    accessor_kind,
    accessor_property_name,
    has_static_method_decorator,
    normalize_gpu_decorator,
    normalize_type_name,
)
from .tokens import KEYWORDS


TOKEN_TYPES = [
    "namespace",
    "type",
    "class",
    "enum",
    "interface",
    "struct",
    "typeParameter",
    "parameter",
    "variable",
    "property",
    "enumMember",
    "event",
    "function",
    "method",
    "macro",
    "keyword",
    "modifier",
    "comment",
    "string",
    "number",
    "regexp",
    "operator",
]
TOKEN_MODIFIERS: list[str] = []
TOKEN_INDEX = {name: i for i, name in enumerate(TOKEN_TYPES)}


BUILTIN_NAMESPACE_COMPLETIONS = ["std", "__intrin"]
BUILTIN_SPECIAL_COMPLETIONS = ["type", "isinstance", "issubclass", "id", "help"]
BUILTIN_COMPLETIONS = sorted([*BUILTIN_NAMESPACE_COMPLETIONS, *BUILTIN_ALIASES.keys(), *BUILTIN_SPECIAL_COMPLETIONS])
INTRINSIC_NAMES = [spec.name for spec in all_intrinsics()]
CUDA_INTRINSIC_PREFIX = "cuda_"


@dataclass
class SymbolEntry:
    name: str
    kind: str
    uri: str
    range: Range
    selection_range: Range
    detail: str = ""
    container: str | None = None
    docstring: str | None = None


@dataclass
class AnalyzedDocument:
    uri: str
    path: Path
    text: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    symbols: list[SymbolEntry] = field(default_factory=list)
    tokens: list[tuple[str, Range]] = field(default_factory=list)


class ManvLanguageServer(LanguageServer):
    def __init__(self) -> None:
        super().__init__(name="manv-lsp", version="0.1.0")
        self.root_uri: str | None = None
        self.documents: dict[str, str] = {}
        self.analysis: dict[str, AnalyzedDocument] = {}

    def analyze_uri(self, uri: str) -> AnalyzedDocument:
        text = self.documents.get(uri)
        if text is None:
            path = _uri_to_path(uri)
            if path.exists():
                text = path.read_text(encoding="utf-8")
                self.documents[uri] = text
            else:
                text = ""

        analyzed = _analyze_document(uri, text)
        self.analysis[uri] = analyzed
        self.publish_diagnostics(uri, analyzed.diagnostics)
        return analyzed

    def get_or_analyze(self, uri: str) -> AnalyzedDocument:
        analyzed = self.analysis.get(uri)
        if analyzed is not None:
            return analyzed
        return self.analyze_uri(uri)

    def all_workspace_documents(self) -> dict[str, str]:
        docs = dict(self.documents)
        root = _uri_to_path(self.root_uri) if self.root_uri else None
        if root and root.exists():
            for path in root.rglob("*.mv"):
                if _is_ignored_path(path):
                    continue
                uri = _path_to_uri(path)
                if uri not in docs:
                    try:
                        docs[uri] = path.read_text(encoding="utf-8")
                    except OSError:
                        continue
        return docs


def create_server() -> ManvLanguageServer:
    server = ManvLanguageServer()

    @server.feature("initialize")
    def on_initialize(ls: ManvLanguageServer, params: InitializeParams) -> InitializeResult:
        ls.root_uri = params.root_uri
        capabilities = ServerCapabilities(
            text_document_sync=TextDocumentSyncKind.Incremental,
            completion_provider=CompletionOptions(trigger_characters=["."]),
            hover_provider=True,
            definition_provider=True,
            references_provider=True,
            rename_provider=True,
            document_symbol_provider=True,
            workspace_symbol_provider=True,
            signature_help_provider=SignatureHelpOptions(trigger_characters=["(", ","]),
            semantic_tokens_provider=SemanticTokensOptions(
                legend=SemanticTokensLegend(token_types=TOKEN_TYPES, token_modifiers=TOKEN_MODIFIERS),
                full=True,
            ),
        )
        return InitializeResult(capabilities=capabilities)

    @server.feature("textDocument/didOpen")
    def did_open(ls: ManvLanguageServer, params: DidOpenTextDocumentParams) -> None:
        uri = params.text_document.uri
        ls.documents[uri] = params.text_document.text
        ls.analyze_uri(uri)

    @server.feature("textDocument/didChange")
    def did_change(ls: ManvLanguageServer, params: DidChangeTextDocumentParams) -> None:
        uri = params.text_document.uri
        current = ls.documents.get(uri, "")
        for change in params.content_changes:
            if change.range is None:
                current = change.text
            else:
                current = _apply_incremental_change(current, change.range, change.text)
        ls.documents[uri] = current
        ls.analyze_uri(uri)

    @server.feature("textDocument/didSave")
    def did_save(ls: ManvLanguageServer, params: DidSaveTextDocumentParams) -> None:
        uri = params.text_document.uri
        if params.text is not None:
            ls.documents[uri] = params.text
        ls.analyze_uri(uri)

    @server.feature("textDocument/didClose")
    def did_close(ls: ManvLanguageServer, params: DidCloseTextDocumentParams) -> None:
        uri = params.text_document.uri
        ls.documents.pop(uri, None)
        ls.analysis.pop(uri, None)
        ls.publish_diagnostics(uri, [])

    @server.feature("textDocument/completion")
    def completion(ls: ManvLanguageServer, params: Any) -> CompletionList:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        line_text = _line_at(analyzed.text, params.position.line)
        prefix = line_text[: params.position.character]

        decorator_items = _decorator_completion_items(prefix)
        if decorator_items is not None:
            return CompletionList(is_incomplete=False, items=decorator_items)

        intrinsic_items = _intrinsic_completion_items(prefix)
        if intrinsic_items is not None:
            return CompletionList(is_incomplete=False, items=intrinsic_items)

        items: list[CompletionItem] = []
        seen_labels: set[str] = set()

        def add_item(item: CompletionItem) -> None:
            if item.label in seen_labels:
                return
            seen_labels.add(item.label)
            items.append(item)

        for kw in sorted(KEYWORDS):
            add_item(CompletionItem(label=kw, kind=CompletionItemKind.Keyword))
        for builtin in BUILTIN_COMPLETIONS:
            kind = CompletionItemKind.Module if builtin in BUILTIN_NAMESPACE_COMPLETIONS else CompletionItemKind.Function
            add_item(CompletionItem(label=builtin, kind=kind, detail=_builtin_completion_detail(builtin)))
        for symbol in analyzed.symbols:
            add_item(
                CompletionItem(
                    label=symbol.name,
                    kind=_completion_kind(symbol.kind),
                    detail=symbol.detail or symbol.kind,
                )
            )

        return CompletionList(is_incomplete=False, items=items)

    @server.feature("textDocument/hover")
    def hover(ls: ManvLanguageServer, params: HoverParams) -> Hover | None:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        decorator_name = _decorator_name_at(analyzed.text, params.position)
        if decorator_name == "gpu":
            value = "**decorator** `@gpu`\n\nMarks a function as GPU-eligible while preserving HLIR-authoritative fallback semantics."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))
        if decorator_name == "static_method":
            value = "**decorator** `@static_method`\n\nMarks a type or impl method as callable on the type object without an instance receiver."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))
        if decorator_name == "getter":
            value = "**decorator** `@getter`\n\nMarks an instance method as the property getter used by `obj.name`."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))
        if decorator_name == "setter":
            value = "**decorator** `@setter`\n\nMarks an instance method as the property setter used by `obj.name = value`."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))

        name = _word_at_position(analyzed.text, params.position)
        if name is None:
            return None

        intrinsic_name = _intrinsic_name_at(analyzed.text, params.position)
        if intrinsic_name:
            spec = resolve_intrinsic(intrinsic_name)
            if spec is not None:
                args = ", ".join(_intrin_type_label(x) for x in spec.arg_types)
                ret = _intrin_type_label(spec.return_type)
                value = f"**intrinsic** `{_intrinsic_public_name(intrinsic_name)}`\n\n`({args}) -> {ret}`"
                return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))

        if name in BUILTIN_ALIASES or name in BUILTIN_SPECIAL_COMPLETIONS:
            builtin_sig = _signature_for_builtin(name)
            if builtin_sig is not None:
                label, _ = builtin_sig
                value = f"**builtin** `{name}`\n\n`{label}`"
                return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))

        namespace_chain = _attribute_chain_at(analyzed.text, params.position)
        if name == "cuda" and namespace_chain is not None and namespace_chain.startswith("__intrin.cuda"):
            value = "**namespace** `__intrin.cuda`\n\nCUDA runtime intrinsics surfaced through the compiler-owned intrinsic bridge."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))
        if name == "__intrin":
            value = "**namespace** `__intrin`\n\nCompiler-owned intrinsic namespace."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))
        if name == "std":
            value = "**namespace** `std`\n\nStandard library root namespace."
            return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value))

        matches = [s for s in analyzed.symbols if s.name == name]
        if not matches:
            matches = _workspace_symbols(ls, name)
        if not matches:
            return None
        sym = _pick_best_symbol(matches, params.position, prefer_uri=uri)
        value = f"**{sym.kind}** `{sym.name}`"
        if sym.detail:
            value += f"\n\n`{sym.detail}`"
        if sym.docstring:
            value += f"\n\n{sym.docstring}"
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=value), range=sym.selection_range)

    @server.feature("textDocument/definition")
    def definition(ls: ManvLanguageServer, params: DefinitionParams) -> Location | list[Location] | None:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        intrinsic_chain = _attribute_chain_at(analyzed.text, params.position)
        if intrinsic_chain is not None and intrinsic_chain.startswith("__intrin."):
            return None
        name = _word_at_position(analyzed.text, params.position)
        if name is None:
            return None
        if name in BUILTIN_ALIASES or name in BUILTIN_NAMESPACE_COMPLETIONS:
            return None

        all_symbols = _workspace_symbols(ls, name)
        if not all_symbols:
            return None

        best = _pick_best_symbol(all_symbols, params.position, prefer_uri=uri)
        return Location(uri=best.uri, range=best.selection_range)

    @server.feature("textDocument/references")
    def references(ls: ManvLanguageServer, params: ReferenceParams) -> list[Location]:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        intrinsic_chain = _attribute_chain_at(analyzed.text, params.position)
        if intrinsic_chain is not None and intrinsic_chain.startswith("__intrin."):
            return []
        name = _word_at_position(analyzed.text, params.position)
        if name is None:
            return []
        if name in BUILTIN_ALIASES or name in BUILTIN_NAMESPACE_COMPLETIONS:
            return []

        include_decl = bool(params.context.include_declaration)
        all_docs = ls.all_workspace_documents()
        locations: list[Location] = []

        definitions = _definition_ranges(ls, name)

        for doc_uri, text in all_docs.items():
            tokens = _identifier_tokens(text, _uri_to_path(doc_uri))
            for token_name, token_range in tokens:
                if token_name != name:
                    continue
                is_decl = any(_same_range(token_range, d) and doc_uri == d_uri for d_uri, d in definitions)
                if is_decl and not include_decl:
                    continue
                locations.append(Location(uri=doc_uri, range=token_range))

        return locations

    @server.feature("textDocument/rename")
    def rename(ls: ManvLanguageServer, params: RenameParams) -> WorkspaceEdit | None:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        intrinsic_chain = _attribute_chain_at(analyzed.text, params.position)
        if (intrinsic_chain is not None and intrinsic_chain.startswith("__intrin.")) or _decorator_name_at(analyzed.text, params.position) is not None:
            return None
        old_name = _word_at_position(analyzed.text, params.position)
        if old_name is None:
            return None
        if old_name in BUILTIN_ALIASES or old_name in BUILTIN_NAMESPACE_COMPLETIONS:
            return None
        if not _is_valid_identifier(params.new_name):
            return None

        edits: dict[str, list[TextEdit]] = {}
        for doc_uri, text in ls.all_workspace_documents().items():
            token_edits: list[TextEdit] = []
            for token_name, token_range in _identifier_tokens(text, _uri_to_path(doc_uri)):
                if token_name == old_name:
                    token_edits.append(TextEdit(range=token_range, new_text=params.new_name))
            if token_edits:
                edits[doc_uri] = token_edits

        if not edits:
            return None
        return WorkspaceEdit(changes=edits)

    @server.feature("textDocument/prepareRename")
    def prepare_rename(ls: ManvLanguageServer, params: PrepareRenameParams) -> Range | None:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        intrinsic_chain = _attribute_chain_at(analyzed.text, params.position)
        if (intrinsic_chain is not None and intrinsic_chain.startswith("__intrin.")) or _decorator_name_at(analyzed.text, params.position) is not None:
            return None
        word = _word_with_range(analyzed.text, params.position)
        if word is None:
            return None
        name, token_range = word
        if name in KEYWORDS or name in {"true", "false", "none"} or name in BUILTIN_ALIASES or name in BUILTIN_NAMESPACE_COMPLETIONS:
            return None
        return token_range

    @server.feature("textDocument/documentSymbol")
    def document_symbol(ls: ManvLanguageServer, params: DocumentSymbolParams) -> list[DocumentSymbol]:
        analyzed = ls.get_or_analyze(params.text_document.uri)
        top: list[DocumentSymbol] = []
        by_container: dict[str, list[DocumentSymbol]] = {}

        for sym in analyzed.symbols:
            ds = DocumentSymbol(
                name=sym.name,
                detail=sym.detail,
                kind=_symbol_kind(sym.kind),
                range=sym.range,
                selection_range=sym.selection_range,
                children=[],
            )
            if sym.container:
                by_container.setdefault(sym.container, []).append(ds)
            else:
                top.append(ds)

        for root in top:
            if root.name in by_container:
                root.children = sorted(by_container[root.name], key=lambda d: (d.range.start.line, d.range.start.character))

        return sorted(top, key=lambda d: (d.range.start.line, d.range.start.character))

    @server.feature("workspace/symbol")
    def workspace_symbol(ls: ManvLanguageServer, params: WorkspaceSymbolParams) -> list[SymbolInformation]:
        query = (params.query or "").strip().lower()
        results: list[SymbolInformation] = []
        for uri, text in ls.all_workspace_documents().items():
            analyzed = ls.analysis.get(uri)
            if analyzed is None:
                analyzed = _analyze_document(uri, text)
                ls.analysis[uri] = analyzed
            for sym in analyzed.symbols:
                if query and query not in sym.name.lower():
                    continue
                results.append(
                    SymbolInformation(
                        name=sym.name,
                        kind=_symbol_kind(sym.kind),
                        location=Location(uri=sym.uri, range=sym.selection_range),
                        container_name=sym.container,
                    )
                )
        return results

    @server.feature("textDocument/signatureHelp")
    def signature_help(ls: ManvLanguageServer, params: Any) -> SignatureHelp | None:
        uri = params.text_document.uri
        analyzed = ls.get_or_analyze(uri)
        call_info = _call_context(analyzed.text, params.position)
        if call_info is None:
            return None
        callee, arg_index = call_info

        sig = _signature_for_name(analyzed, callee)
        if sig is None:
            return None
        signature, params_labels = sig
        return SignatureHelp(
            signatures=[
                SignatureInformation(
                    label=signature,
                    parameters=[ParameterInformation(label=lbl) for lbl in params_labels],
                )
            ],
            active_signature=0,
            active_parameter=max(0, arg_index),
        )

    @server.feature("textDocument/semanticTokens/full")
    def semantic_tokens(ls: ManvLanguageServer, params: SemanticTokensParams) -> SemanticTokens:
        analyzed = ls.get_or_analyze(params.text_document.uri)
        encoded = _encode_semantic_tokens(analyzed)
        return SemanticTokens(data=encoded)

    return server


def start_stdio() -> None:
    server = create_server()
    server.start_io()


def start_tcp(host: str = "127.0.0.1", port: int = 2087) -> None:
    server = create_server()
    server.start_tcp(host=host, port=port)


def _analyze_document(uri: str, text: str) -> AnalyzedDocument:
    path = _uri_to_path(uri)
    diagnostics: list[Diagnostic] = []
    symbols: list[SymbolEntry] = []

    try:
        program = parse_program(text, str(path))
        symbols = _symbols_from_program(uri, program)
        analyzer = SemanticAnalyzer(str(path))
        result = analyzer.analyze(program)
        diagnostics.extend(_to_lsp_diagnostic(d) for d in result.diagnostics)
    except ManvError as err:
        diagnostics.extend(_to_lsp_diagnostic(d) for d in [err.diagnostic, *err.extra])

    tokens = _identifier_tokens(text, path)
    return AnalyzedDocument(uri=uri, path=path, text=text, diagnostics=diagnostics, symbols=symbols, tokens=tokens)


def _to_lsp_diagnostic(d: ManvDiagnostic) -> Diagnostic:
    sev = DiagnosticSeverity.Error
    if d.severity == "warning":
        sev = DiagnosticSeverity.Warning
    elif d.severity == "information":
        sev = DiagnosticSeverity.Information
    elif d.severity == "hint":
        sev = DiagnosticSeverity.Hint

    start = Position(line=max(0, d.span.line - 1), character=max(0, d.span.column - 1))
    end_line = max(0, (d.span.end_line - 1) if d.span.end_line else start.line)
    end_col = max(start.character + 1, (d.span.end_column - 1) if d.span.end_column else (start.character + 1))
    end = Position(line=end_line, character=end_col)

    return Diagnostic(range=Range(start=start, end=end), severity=sev, code=d.code, source="manv", message=d.message)


def _symbols_from_program(uri: str, program: ast.Program) -> list[SymbolEntry]:
    out: list[SymbolEntry] = []

    for decl in program.declarations:
        if isinstance(decl, ast.FnDecl):
            fn_sym = _fn_symbol(uri, decl, container=None)
            out.append(fn_sym)
            out.extend(_param_symbols(uri, decl, container=decl.name))
            out.extend(_stmt_symbols(uri, decl.body, container=decl.name))
        elif isinstance(decl, ast.TypeDecl):
            type_range = _name_range(decl.span, decl.name)
            out.append(
                SymbolEntry(
                    name=decl.name,
                    kind="type",
                    uri=uri,
                    range=type_range,
                    selection_range=type_range,
                    detail=f"type {decl.name}",
                    container=None,
                    docstring=decl.docstring,
                )
            )
            for attr in decl.attrs:
                rng = _name_range(attr.span, attr.name)
                out.append(
                    SymbolEntry(
                        name=attr.name,
                        kind="field",
                        uri=uri,
                        range=rng,
                        selection_range=rng,
                        detail=f"const {attr.name}: {normalize_type_name(attr.type_name) or 'any'}",
                        container=decl.name,
                    )
                )
            out.extend(_type_member_symbols(uri, decl.name, decl.methods))
        elif isinstance(decl, ast.ImplDecl):
            out.extend(_type_member_symbols(uri, decl.target, decl.methods))

    out.extend(_stmt_symbols(uri, program.statements, container=None))
    return out


def _fn_symbol(uri: str, fn: ast.FnDecl, *, container: str | None, kind: str = "function") -> SymbolEntry:
    sig = _fn_signature(fn)
    rng = _name_range(fn.span, fn.name)
    return SymbolEntry(
        name=fn.name,
        kind=kind,
        uri=uri,
        range=rng,
        selection_range=rng,
        detail=sig,
        container=container,
        docstring=fn.docstring,
    )


def _type_member_symbols(uri: str, owner: str, methods: list[ast.FnDecl]) -> list[SymbolEntry]:
    """Render ordinary methods and accessor-backed properties for one type."""

    out: list[SymbolEntry] = []
    properties: dict[str, dict[str, ast.FnDecl]] = {}

    for method in methods:
        accessor = accessor_kind(method)
        if accessor in {"getter", "setter"}:
            try:
                property_name = accessor_property_name(method) or method.name
            except Exception:
                property_name = method.name
            properties.setdefault(property_name, {})[accessor] = method
            continue
        out.append(_fn_symbol(uri, method, container=owner, kind="method"))
        out.extend(_param_symbols(uri, method, container=method.name))
        out.extend(_stmt_symbols(uri, method.body, container=method.name))

    for property_name, accessors in sorted(properties.items()):
        representative = accessors.get("getter") or accessors.get("setter")
        if representative is None:
            continue
        available = []
        if "getter" in accessors:
            available.append("get")
        if "setter" in accessors:
            available.append("set")
        rng = _name_range(representative.span, representative.name)
        out.append(
            SymbolEntry(
                name=property_name,
                kind="field",
                uri=uri,
                range=rng,
                selection_range=rng,
                detail=f"property {property_name} ({', '.join(available)})",
                container=owner,
                docstring=representative.docstring,
            )
        )
    return out


def _param_symbols(uri: str, fn: ast.FnDecl, *, container: str) -> list[SymbolEntry]:
    out: list[SymbolEntry] = []
    for p in fn.params:
        rng = _name_range(p.span, p.name)
        out.append(
            SymbolEntry(
                name=p.name,
                kind="parameter",
                uri=uri,
                range=rng,
                selection_range=rng,
                detail=f"param {p.name}: {normalize_type_name(p.type_name) or 'any'}",
                container=container,
            )
        )
    return out


def _stmt_symbols(uri: str, statements: list[Any], *, container: str | None) -> list[SymbolEntry]:
    out: list[SymbolEntry] = []
    for stmt in statements:
        if isinstance(stmt, ast.LetStmt):
            rng = _name_range(stmt.span, stmt.name)
            out.append(
                SymbolEntry(
                    name=stmt.name,
                    kind="variable",
                    uri=uri,
                    range=rng,
                    selection_range=rng,
                    detail=f"let {stmt.name}: {normalize_type_name(stmt.type_name) or 'any'}",
                    container=container,
                )
            )
        elif isinstance(stmt, ast.IfStmt):
            out.extend(_stmt_symbols(uri, stmt.then_body, container=container))
            out.extend(_stmt_symbols(uri, stmt.else_body, container=container))
        elif isinstance(stmt, ast.WhileStmt):
            out.extend(_stmt_symbols(uri, stmt.body, container=container))
        elif isinstance(stmt, ast.ForStmt):
            rng = _for_var_range(stmt)
            out.append(
                SymbolEntry(
                    name=stmt.var_name,
                    kind="variable",
                    uri=uri,
                    range=rng,
                    selection_range=rng,
                    detail=f"for {stmt.var_name}: i32",
                    container=container,
                )
            )
            out.extend(_stmt_symbols(uri, stmt.body, container=container))
        elif isinstance(stmt, ast.TryStmt):
            out.extend(_stmt_symbols(uri, stmt.try_body, container=container))
            out.extend(_stmt_symbols(uri, stmt.else_body, container=container))
            out.extend(_stmt_symbols(uri, stmt.finally_body, container=container))
            for clause in stmt.except_clauses:
                if clause.bind_name:
                    rng = _name_range(clause.span, clause.bind_name)
                    out.append(
                        SymbolEntry(
                            name=clause.bind_name,
                            kind="variable",
                            uri=uri,
                            range=rng,
                            selection_range=rng,
                            detail=f"except {clause.type_name} as {clause.bind_name}",
                            container=container,
                        )
                    )
                out.extend(_stmt_symbols(uri, clause.body, container=container))
    return out


def _fn_signature(fn: ast.FnDecl) -> str:
    args = ", ".join(f"{p.name}: {normalize_type_name(p.type_name) or 'any'}" for p in fn.params)
    ret = normalize_type_name(fn.return_type) or "none"

    # Hover/document symbol output should reflect the normalized GPU policy seen
    # by lowering, not the raw decorator syntax text.
    try:
        gpu = normalize_gpu_decorator(fn)
    except Exception:
        gpu = None

    prefixes: list[str] = []
    if has_static_method_decorator(fn):
        prefixes.append("@static_method")
    accessor = accessor_kind(fn)
    if accessor == "getter":
        try:
            property_name = accessor_property_name(fn) or fn.name
        except Exception:
            property_name = fn.name
        prefixes.append(f"@getter(name=\"{property_name}\")" if property_name != fn.name else "@getter")
    elif accessor == "setter":
        try:
            property_name = accessor_property_name(fn) or fn.name
        except Exception:
            property_name = fn.name
        prefixes.append(f"@setter(name=\"{property_name}\")" if property_name != fn.name else "@setter")
    if gpu is not None:
        required = "true" if gpu.required else "false"
        prefixes.append(f'@gpu(required={required}, mode="{gpu.mode}")')
    prefix = ""
    if prefixes:
        prefix = " ".join(prefixes) + " "
    return f"{prefix}fn {fn.name}({args}) -> {ret}"


def _intrin_type_label(value: Any) -> str:
    return getattr(value, "name", str(value))


def _name_range(span: Any, name: str) -> Range:
    start = Position(line=max(0, int(getattr(span, "line", 1)) - 1), character=max(0, int(getattr(span, "column", 1)) - 1))
    end = Position(line=start.line, character=start.character + max(1, len(name)))
    return Range(start=start, end=end)


def _completion_kind(kind: str) -> CompletionItemKind:
    if kind in {"function", "method"}:
        return CompletionItemKind.Function
    if kind == "type":
        return CompletionItemKind.Class
    if kind == "field":
        return CompletionItemKind.Property
    if kind == "parameter":
        return CompletionItemKind.Variable
    return CompletionItemKind.Variable


def _symbol_kind(kind: str) -> SymbolKind:
    if kind == "function":
        return SymbolKind.Function
    if kind == "method":
        return SymbolKind.Method
    if kind == "type":
        return SymbolKind.Class
    if kind == "field":
        return SymbolKind.Property
    if kind == "parameter":
        return SymbolKind.Variable
    return SymbolKind.Variable


def _uri_to_path(uri: str | None) -> Path:
    if not uri:
        return Path(".").resolve()
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        path = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return Path(path).resolve()
    return Path(uri).resolve()


def _path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _apply_incremental_change(text: str, rng: Range, new_text: str) -> str:
    lines = text.splitlines(keepends=True)
    start_offset = _position_to_offset(lines, rng.start)
    end_offset = _position_to_offset(lines, rng.end)
    flat = "".join(lines)
    return flat[:start_offset] + new_text + flat[end_offset:]


def _position_to_offset(lines: list[str], pos: Position) -> int:
    line = max(0, pos.line)
    char = max(0, pos.character)
    if not lines:
        return 0
    if line >= len(lines):
        return len("".join(lines))
    offset = sum(len(lines[i]) for i in range(line))
    line_text = lines[line]
    return offset + min(char, len(line_text))


def _line_at(text: str, line: int) -> str:
    lines = text.splitlines()
    if 0 <= line < len(lines):
        return lines[line]
    return ""


def _identifier_tokens(text: str, path: Path) -> list[tuple[str, Range]]:
    try:
        lexer = Lexer(source=text, file=str(path))
        out: list[tuple[str, Range]] = []
        for token in lexer.tokenize():
            if token.kind != "IDENT":
                continue
            start = Position(line=max(0, token.line - 1), character=max(0, token.column - 1))
            end = Position(line=start.line, character=start.character + len(token.lexeme))
            out.append((token.lexeme, Range(start=start, end=end)))
        return out
    except Exception:
        return []


def _word_with_range(text: str, pos: Position) -> tuple[str, Range] | None:
    line_text = _line_at(text, pos.line)
    if not line_text:
        return None
    idx = min(max(0, pos.character), len(line_text))

    left = idx
    while left > 0 and (line_text[left - 1].isalnum() or line_text[left - 1] == "_"):
        left -= 1
    right = idx
    while right < len(line_text) and (line_text[right].isalnum() or line_text[right] == "_"):
        right += 1

    if left == right:
        return None
    name = line_text[left:right]
    rng = Range(
        start=Position(line=pos.line, character=left),
        end=Position(line=pos.line, character=right),
    )
    return name, rng


def _word_at_position(text: str, pos: Position) -> str | None:
    found = _word_with_range(text, pos)
    if found is None:
        return None
    return found[0]


def _intrinsic_name_at(text: str, pos: Position) -> str | None:
    return _resolve_public_intrinsic_name(_attribute_chain_at(text, pos))


def _pick_best_symbol(symbols: list[SymbolEntry], pos: Position, prefer_uri: str | None = None) -> SymbolEntry:
    candidates = symbols
    if prefer_uri is not None:
        same_uri = [s for s in symbols if s.uri == prefer_uri]
        if same_uri:
            candidates = same_uri

    before = [s for s in candidates if (s.selection_range.start.line < pos.line) or (s.selection_range.start.line == pos.line and s.selection_range.start.character <= pos.character)]
    if before:
        return sorted(before, key=lambda s: (s.selection_range.start.line, s.selection_range.start.character), reverse=True)[0]
    return sorted(candidates, key=lambda s: (s.selection_range.start.line, s.selection_range.start.character))[0]


def _definition_ranges(ls: ManvLanguageServer, name: str) -> list[tuple[str, Range]]:
    out: list[tuple[str, Range]] = []
    for doc in _workspace_analyzed_documents(ls).values():
        for sym in doc.symbols:
            if sym.name == name:
                out.append((sym.uri, sym.selection_range))
    return out


def _same_range(a: Range, b: Range) -> bool:
    return (
        a.start.line == b.start.line
        and a.start.character == b.start.character
        and a.end.line == b.end.line
        and a.end.character == b.end.character
    )


def _is_valid_identifier(name: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None and name not in KEYWORDS


def _call_context(text: str, pos: Position) -> tuple[str, int] | None:
    line = _line_at(text, pos.line)
    if not line:
        return None
    prefix = line[: min(len(line), max(0, pos.character))]

    open_idx = prefix.rfind("(")
    if open_idx < 0:
        return None

    head = prefix[:open_idx].rstrip()
    m = re.search(r"([A-Za-z_][A-Za-z0-9_\.]+)$", head)
    if not m:
        return None
    callee = m.group(1)

    args_fragment = prefix[open_idx + 1 :]
    arg_index = 0 if not args_fragment.strip() else args_fragment.count(",")
    return callee, arg_index


def _signature_for_name(analyzed: AnalyzedDocument, callee: str) -> tuple[str, list[str]] | None:
    intrinsic_name = _resolve_public_intrinsic_name(callee)
    if intrinsic_name is not None:
        spec = resolve_intrinsic(intrinsic_name)
        if spec is None:
            return None
        labels = [_intrin_type_label(t) for t in spec.arg_types]
        sig = f"{_intrinsic_public_name(intrinsic_name)}({', '.join(labels)}) -> {_intrin_type_label(spec.return_type)}"
        return sig, labels

    direct = [s for s in analyzed.symbols if s.name == callee and s.kind in {"function", "method"}]
    if direct:
        sym = direct[0]
        label = sym.detail or callee
        params = []
        if "(" in label and ")" in label:
            inside = label.split("(", 1)[1].split(")", 1)[0]
            params = [x.strip() for x in inside.split(",") if x.strip()]
        return label, params

    builtin_sig = _signature_for_builtin(callee)
    if builtin_sig is not None:
        return builtin_sig

    return None


def _encode_semantic_tokens(analyzed: AnalyzedDocument) -> list[int]:
    entries: list[tuple[int, int, int, int, int]] = []

    by_name_kind: dict[str, str] = {}
    for sym in analyzed.symbols:
        by_name_kind.setdefault(sym.name, sym.kind)

    try:
        lexer = Lexer(source=analyzed.text, file=str(analyzed.path))
        tokens = lexer.tokenize()
    except Exception:
        return []

    for idx, token in enumerate(tokens):
        if token.kind in {"NEWLINE", "INDENT", "DEDENT", "EOF"}:
            continue
        line = max(0, token.line - 1)
        start = max(0, token.column - 1)
        length = max(1, len(token.lexeme))

        token_type = TOKEN_INDEX["variable"]
        if token.kind == "KEYWORD":
            token_type = TOKEN_INDEX["modifier"] if _token_has_decorator_marker(tokens, idx) else TOKEN_INDEX["keyword"]
        elif token.kind == "STRING":
            token_type = TOKEN_INDEX["string"]
        elif token.kind == "NUMBER":
            token_type = TOKEN_INDEX["number"]
        elif token.kind == "OP":
            token_type = TOKEN_INDEX["operator"]
        elif token.kind == "IDENT":
            token_type = _identifier_semantic_token_type(tokens, idx, token.lexeme, by_name_kind)

        entries.append((line, start, length, token_type, 0))

    entries.sort(key=lambda x: (x[0], x[1]))
    packed: list[int] = []
    prev_line = 0
    prev_start = 0
    for line, start, length, token_type, mods in entries:
        delta_line = line - prev_line
        delta_start = start - prev_start if delta_line == 0 else start
        packed.extend([delta_line, delta_start, length, token_type, mods])
        prev_line = line
        prev_start = start
    return packed


def _for_var_range(stmt: ast.ForStmt) -> Range:
    start = Position(line=max(0, stmt.span.line - 1), character=max(0, stmt.span.column - 1 + len("for ")))
    end = Position(line=start.line, character=start.character + max(1, len(stmt.var_name)))
    return Range(start=start, end=end)


def _workspace_analyzed_documents(ls: ManvLanguageServer) -> dict[str, AnalyzedDocument]:
    docs = dict(ls.analysis)
    for uri, text in ls.all_workspace_documents().items():
        if uri in docs:
            continue
        analyzed = _analyze_document(uri, text)
        ls.analysis[uri] = analyzed
        docs[uri] = analyzed
    return docs


def _workspace_symbols(ls: ManvLanguageServer, name: str) -> list[SymbolEntry]:
    out: list[SymbolEntry] = []
    for analyzed in _workspace_analyzed_documents(ls).values():
        out.extend([symbol for symbol in analyzed.symbols if symbol.name == name])
    return out


def _intrinsic_public_name(internal_name: str) -> str:
    if internal_name.startswith(CUDA_INTRINSIC_PREFIX):
        return "__intrin.cuda." + internal_name[len(CUDA_INTRINSIC_PREFIX) :]
    return "__intrin." + internal_name


def _resolve_public_intrinsic_name(public_name: str | None) -> str | None:
    if not public_name or not public_name.startswith("__intrin."):
        return None
    remainder = public_name[len("__intrin.") :]
    if not remainder:
        return None
    if remainder.startswith("cuda."):
        leaf = remainder[len("cuda.") :]
        return f"{CUDA_INTRINSIC_PREFIX}{leaf}" if leaf else None
    return remainder


def _builtin_completion_detail(name: str) -> str:
    if name == "std":
        return "standard library namespace"
    if name == "__intrin":
        return "compiler intrinsic namespace"
    builtin_sig = _signature_for_builtin(name)
    return builtin_sig[0] if builtin_sig is not None else "builtin"


def _signature_for_builtin(name: str) -> tuple[str, list[str]] | None:
    if name == "type":
        return "type(value) -> type", ["value"]
    if name == "isinstance":
        return "isinstance(value, cls) -> bool", ["value", "cls"]
    if name == "issubclass":
        return "issubclass(child, parent) -> bool", ["child", "parent"]
    if name == "id":
        return "id(value) -> int", ["value"]
    if name == "help":
        return "help(value) -> none", ["value"]
    intrinsic_name = BUILTIN_ALIASES.get(name)
    if intrinsic_name is None:
        return None
    spec = resolve_intrinsic(intrinsic_name)
    if spec is None:
        return None
    labels = [_intrin_type_label(value) for value in spec.arg_types]
    return f"{name}({', '.join(labels)}) -> {_intrin_type_label(spec.return_type)}", labels


def _decorator_completion_items(prefix: str) -> list[CompletionItem] | None:
    accessor_name_match = re.search(r"@(getter|setter)\(([^)]*)\bname\s*=\s*\"([A-Za-z_]*)$", prefix)
    if accessor_name_match is not None:
        return []

    accessor_kw_scope = re.search(r"@(getter|setter)\(([^)]*)$", prefix)
    if accessor_kw_scope is not None:
        inner = accessor_kw_scope.group(2).rsplit(",", 1)[-1].rsplit("(", 1)[-1].strip()
        if inner and "=" not in inner and '"' not in inner and "'" not in inner:
            return [
                CompletionItem(label="name", kind=CompletionItemKind.Property, detail="string literal property name")
            ]

    mode_unquoted = re.search(r"@gpu\([^)]*\bmode\s*=\s*([A-Za-z_]*)$", prefix)
    if mode_unquoted is not None:
        partial = mode_unquoted.group(1)
        return [
            CompletionItem(label=f'"{value}"', kind=CompletionItemKind.Value, detail="gpu mode")
            for value in ("kernel", "graph")
            if value.startswith(partial)
        ]

    mode_quoted = re.search(r'@gpu\([^)]*\bmode\s*=\s*"([A-Za-z_]*)$', prefix)
    if mode_quoted is not None:
        partial = mode_quoted.group(1)
        return [
            CompletionItem(label=value, kind=CompletionItemKind.Value, detail="gpu mode")
            for value in ("kernel", "graph")
            if value.startswith(partial)
        ]

    required_match = re.search(r"@gpu\([^)]*\brequired\s*=\s*([A-Za-z_]*)$", prefix)
    if required_match is not None:
        partial = required_match.group(1)
        return [
            CompletionItem(label=value, kind=CompletionItemKind.Keyword, detail="gpu required flag")
            for value in ("true", "false", "True", "False")
            if value.startswith(partial)
        ]

    kwarg_scope = re.search(r"@gpu\([^)]*$", prefix)
    if kwarg_scope is not None:
        tail = kwarg_scope.group(0).rsplit(",", 1)[-1].rsplit("(", 1)[-1].strip()
        if tail and "=" not in tail and '"' not in tail and "'" not in tail:
            return [
                item
                for item in [
                    CompletionItem(label="required", kind=CompletionItemKind.Property, detail="bool literal"),
                    CompletionItem(label="mode", kind=CompletionItemKind.Property, detail='string literal: "kernel" | "graph"'),
                ]
                if item.label.startswith(tail)
            ]

    decorator_match = re.search(r"@([A-Za-z_]*)$", prefix)
    if decorator_match is not None:
        partial = decorator_match.group(1)
        items: list[CompletionItem] = []
        if "gpu".startswith(partial):
            items.append(CompletionItem(label="gpu", kind=CompletionItemKind.Function, detail="GPU-eligible function decorator"))
        if "static_method".startswith(partial):
            items.append(CompletionItem(label="static_method", kind=CompletionItemKind.Function, detail="Type-callable method decorator"))
        if "getter".startswith(partial):
            items.append(CompletionItem(label="getter", kind=CompletionItemKind.Function, detail="Instance property getter decorator"))
        if "setter".startswith(partial):
            items.append(CompletionItem(label="setter", kind=CompletionItemKind.Function, detail="Instance property setter decorator"))
        return items
    return None


def _intrinsic_completion_items(prefix: str) -> list[CompletionItem] | None:
    cuda_match = re.search(r"__intrin\.cuda\.([A-Za-z_0-9]*)$", prefix)
    if cuda_match is not None:
        partial = cuda_match.group(1)
        items: list[CompletionItem] = []
        for name in sorted(INTRINSIC_NAMES):
            if not name.startswith(CUDA_INTRINSIC_PREFIX):
                continue
            label = name[len(CUDA_INTRINSIC_PREFIX) :]
            if not label.startswith(partial):
                continue
            items.append(
                CompletionItem(
                    label=label,
                    kind=CompletionItemKind.Function,
                    detail=_intrinsic_signature_detail(name),
                )
            )
        return items

    root_match = re.search(r"__intrin\.([A-Za-z_0-9]*)$", prefix)
    if root_match is None:
        return None

    partial = root_match.group(1)
    items: list[CompletionItem] = []
    if "cuda".startswith(partial):
        items.append(CompletionItem(label="cuda", kind=CompletionItemKind.Module, detail="CUDA intrinsic namespace"))
    for name in sorted(INTRINSIC_NAMES):
        if name.startswith(CUDA_INTRINSIC_PREFIX):
            continue
        if not name.startswith(partial):
            continue
        items.append(
            CompletionItem(
                label=name,
                kind=CompletionItemKind.Function,
                detail=_intrinsic_signature_detail(name),
            )
        )
    return items


def _intrinsic_signature_detail(name: str) -> str:
    spec = resolve_intrinsic(name)
    if spec is None:
        return "intrinsic"
    args = ", ".join(_intrin_type_label(value) for value in spec.arg_types)
    return f"{_intrinsic_public_name(name)}({args}) -> {_intrin_type_label(spec.return_type)}"


def _decorator_name_at(text: str, pos: Position) -> str | None:
    word = _word_with_range(text, pos)
    if word is None:
        return None
    name, token_range = word
    line_text = _line_at(text, pos.line)
    if token_range.start.character > 0 and line_text[token_range.start.character - 1] == "@":
        return name
    return None


def _attribute_chain_at(text: str, pos: Position) -> str | None:
    line_text = _line_at(text, pos.line)
    if not line_text:
        return None

    target = min(max(0, pos.character), len(line_text))
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", line_text):
        start, end = match.span()
        if start <= target <= end:
            return match.group(0)
        if target > 0 and start <= target - 1 < end:
            return match.group(0)
    return None


def _prev_significant_index(tokens: list[Any], index: int) -> int | None:
    cursor = index - 1
    while cursor >= 0:
        if tokens[cursor].kind not in {"NEWLINE", "INDENT", "DEDENT"}:
            return cursor
        cursor -= 1
    return None


def _next_significant_index(tokens: list[Any], index: int) -> int | None:
    cursor = index + 1
    while cursor < len(tokens):
        if tokens[cursor].kind not in {"NEWLINE", "INDENT", "DEDENT"}:
            return cursor
        cursor += 1
    return None


def _token_has_decorator_marker(tokens: list[Any], index: int) -> bool:
    prev_idx = _prev_significant_index(tokens, index)
    return prev_idx is not None and tokens[prev_idx].kind == "OP" and tokens[prev_idx].lexeme == "@"


def _identifier_semantic_token_type(
    tokens: list[Any],
    index: int,
    lexeme: str,
    by_name_kind: dict[str, str],
) -> int:
    if lexeme in {"std", "__intrin"}:
        return TOKEN_INDEX["namespace"]

    prev_idx = _prev_significant_index(tokens, index)
    next_idx = _next_significant_index(tokens, index)
    prev_prev_idx = _prev_significant_index(tokens, prev_idx) if prev_idx is not None else None

    if lexeme in BUILTIN_ALIASES and next_idx is not None and tokens[next_idx].kind == "OP" and tokens[next_idx].lexeme == "(":
        return TOKEN_INDEX["function"]
    if (
        prev_idx is not None
        and tokens[prev_idx].kind == "OP"
        and tokens[prev_idx].lexeme == "."
        and prev_prev_idx is not None
        and tokens[prev_prev_idx].lexeme == "__intrin"
        and lexeme == "cuda"
    ):
        return TOKEN_INDEX["namespace"]
    if _is_intrinsic_leaf_token(tokens, index):
        return TOKEN_INDEX["function"]
    if next_idx is not None and tokens[next_idx].kind == "OP" and tokens[next_idx].lexeme == "(":
        return TOKEN_INDEX["function"]

    kind = by_name_kind.get(lexeme)
    if kind == "function":
        return TOKEN_INDEX["function"]
    if kind == "method":
        return TOKEN_INDEX["method"]
    if kind == "type":
        return TOKEN_INDEX["class"]
    if kind == "parameter":
        return TOKEN_INDEX["parameter"]
    return TOKEN_INDEX["variable"]


def _is_intrinsic_leaf_token(tokens: list[Any], index: int) -> bool:
    parts = [tokens[index].lexeme]
    cursor = index
    while True:
        dot_idx = _prev_significant_index(tokens, cursor)
        if dot_idx is None or tokens[dot_idx].kind != "OP" or tokens[dot_idx].lexeme != ".":
            break
        head_idx = _prev_significant_index(tokens, dot_idx)
        if head_idx is None or tokens[head_idx].kind != "IDENT":
            break
        parts.append(tokens[head_idx].lexeme)
        cursor = head_idx
    parts.reverse()
    return bool(parts and parts[0] == "__intrin" and parts[-1] not in {"__intrin", "cuda"})


def _is_ignored_path(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return bool(parts.intersection({".git", ".manv", "dist", "__pycache__", ".pytest_cache"}))
