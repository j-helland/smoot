use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, GenericParam, Generics};

#[proc_macro_derive(Hashable)]
pub fn derive_hashable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    let fields = match input.data {
        Data::Struct(data) => data.fields,
        _ => {
            return syn::Error::new_spanned(input, "Hashable can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let hash_calls = fields.iter().enumerate().map(|(i, field)| {
        if let Some(ident) = &field.ident {
            // Named fields
            quote! { self.#ident.hash(hasher); }
        } else {
            // Unnamed (tuple) fields e.g. `self.0.hash`
            let index = syn::Index::from(i);
            quote! { self.#index.hash(hasher); }
        }
    });

    let generics = add_trait_bounds(input.generics.clone());
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics Hash for #name #type_generics #where_clause {
            fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
                #(#hash_calls)*
            }
        }
    };

    TokenStream::from(expanded)
}

fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            type_param.bounds.push(syn::parse_quote!(Hash));
        }
    }
    generics
}
